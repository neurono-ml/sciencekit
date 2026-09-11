//! A generic prefetch streaming driver (spec `streaming-executor`).
//!
//! Consumes a sequential [`SKLazySource`] through a dedicated I/O thread into a
//! bounded double buffer, handing each owned [`SKDataBatch`] to an
//! algorithm-supplied `update(batch, parallelism, state)` step that runs on the
//! compute side. Input and computation overlap (the reader fetches batch k+1
//! while batch k computes), bounded by the plan's buffer depth; batches are
//! delivered in order, fully owned.

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};

use super::plan::SKExecutionPlan;
use crate::SKError;
use crate::SKFloat;
use crate::batching::{SKDataBatch, SKLazySource};

/// The decision an update step returns to steer the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKStreamDecision {
    /// Keep reading and computing the next batch.
    Continue,
    /// Stop the pipeline: keep the current batch's effect, read no further.
    Stop,
}

/// A bounded, multi-producer/single-consumer item channel.
enum Item<F> {
    /// An owned batch ready for computation.
    Batch(SKDataBatch<F>),
    /// The producer reached the end of the finite source.
    End,
    /// The producer hit a fallible read error.
    Error(SKError),
}

struct Shared<F> {
    inner: Mutex<Inner<F>>,
    condition: Condvar,
}

struct Inner<F> {
    queue: VecDeque<Item<F>>,
    capacity: usize,
    done: bool,
    stop: bool,
}

impl<F> Shared<F> {
    fn new(capacity: usize) -> Self {
        Shared {
            inner: Mutex::new(Inner {
                queue: VecDeque::new(),
                capacity: capacity.max(1),
                done: false,
                stop: false,
            }),
            condition: Condvar::new(),
        }
    }
}

/// The I/O-thread producer loop: read batches from the source into the bounded
/// buffer, stopping early when the consumer requests it. Returns the read error
/// (converted to the central taxonomy) if one occurs.
fn run_producer<F, Source>(source: &mut Source, shared: &Shared<F>) -> Result<(), SKError>
where
    F: SKFloat + Send,
    Source: SKLazySource<F> + Send,
    Source::Error: From<SKError> + Into<SKError>,
{
    loop {
        // Wait for space (or a stop/done signal) before reading.
        {
            let mut guard = shared.inner.lock().unwrap();
            while !guard.stop && !guard.done && guard.queue.len() >= guard.capacity {
                guard = shared.condition.wait(guard).unwrap();
            }
            if guard.stop || guard.done {
                return Ok(());
            }
        }

        let item = match source.next_batch() {
            Ok(Some(batch)) => Item::Batch(batch),
            Ok(None) => Item::End,
            Err(error) => Item::Error(error.into()),
        };

        // Push, waiting for space again if the buffer filled while reading.
        let mut guard = shared.inner.lock().unwrap();
        while !guard.stop && !guard.done && guard.queue.len() >= guard.capacity {
            guard = shared.condition.wait(guard).unwrap();
        }
        if guard.stop {
            return Ok(());
        }
        let is_terminal = matches!(item, Item::End | Item::Error(_));
        guard.queue.push_back(item);
        if is_terminal {
            guard.done = true;
        }
        shared.condition.notify_all();
        if is_terminal {
            return Ok(());
        }
    }
}

/// The consumer: pop the next item, blocking until the producer delivers.
fn next_item<F>(shared: &Shared<F>) -> Option<Item<F>> {
    let mut guard = shared.inner.lock().unwrap();
    loop {
        if guard.stop && guard.queue.is_empty() {
            return None;
        }
        while guard.queue.is_empty() && !guard.done {
            guard = shared.condition.wait(guard).unwrap();
        }
        if let Some(item) = guard.queue.pop_front() {
            shared.condition.notify_all();
            return Some(item);
        }
        if guard.done {
            return None;
        }
    }
}

/// Request the producer to stop reading further batches.
fn request_stop<F>(shared: &Shared<F>) {
    let mut guard = shared.inner.lock().unwrap();
    guard.stop = true;
    shared.condition.notify_all();
}

/// Run the prefetch pipeline over a [`SKLazySource`], applying `update` to each
/// owned batch in source order.
///
/// A dedicated I/O thread reads ahead into a bounded buffer of the plan's
/// [`SKExecutionPlan::buffer_depth`]; the compute side runs `update` on the
/// caller's thread, receiving the batch, the plan's `parallelism` and the
/// mutable algorithm state. `update` returns [`SKStreamDecision::Continue`] to
/// keep going or [`SKStreamDecision::Stop`] to halt the pipeline, keeping the
/// current batch's effect. A mid-stream read failure surfaces as the central
/// taxonomy error with the effects of earlier batches preserved.
pub fn sk_run_streaming_driver<F, Source, State, Update>(
    source: &mut Source,
    state: &mut State,
    update: Update,
    plan: &SKExecutionPlan,
) -> Result<(), SKError>
where
    F: SKFloat + Send,
    Source: SKLazySource<F> + Send,
    Source::Error: From<SKError> + Into<SKError>,
    Update: Fn(SKDataBatch<F>, usize, &mut State) -> SKStreamDecision,
{
    let shared = Arc::new(Shared::new(plan.buffer_depth));
    std::thread::scope(|scope| {
        let producer_shared = Arc::clone(&shared);
        let handle = scope.spawn(move || run_producer(source, &producer_shared));

        let mut result = Ok(());
        loop {
            match next_item(&shared) {
                Some(Item::Batch(batch)) => match update(batch, plan.parallelism, state) {
                    SKStreamDecision::Continue => continue,
                    SKStreamDecision::Stop => {
                        request_stop(&shared);
                        break;
                    }
                },
                Some(Item::Error(error)) => {
                    request_stop(&shared);
                    result = Err(error);
                    break;
                }
                Some(Item::End) | None => break,
            }
        }
        // Ensure the producer thread exits before the scope closes.
        let _ = handle.join();
        result
    })
}
