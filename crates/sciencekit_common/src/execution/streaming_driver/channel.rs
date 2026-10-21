//! The bounded item channel shared by the producer and the consumer threads.

use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};

use crate::SKError;

/// A bounded, multi-producer/single-consumer item channel.
pub(super) enum Item<T> {
    /// An owned batch ready for computation.
    Batch(T),
    /// The producer reached the end of the finite source.
    End,
    /// The producer hit a fallible read error.
    Error(SKError),
}

pub(super) struct Shared<T> {
    inner: Mutex<Inner<T>>,
    condition: Condvar,
}

struct Inner<T> {
    queue: VecDeque<Item<T>>,
    capacity: usize,
    done: bool,
    stop: bool,
}

impl<T> Shared<T> {
    pub(super) fn new(capacity: usize) -> Self {
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

/// The I/O-thread producer loop: read batches through `read` into the bounded
/// buffer, stopping early when the consumer requests it.
pub(super) fn run_producer<T, Read>(mut read: Read, shared: &Shared<T>) -> Result<(), SKError>
where
    T: Send,
    Read: FnMut() -> Result<Option<T>, SKError>,
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

        let item = match read() {
            Ok(Some(batch)) => Item::Batch(batch),
            Ok(None) => Item::End,
            Err(error) => Item::Error(error),
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
pub(super) fn next_item<T>(shared: &Shared<T>) -> Option<Item<T>> {
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
pub(super) fn request_stop<T>(shared: &Shared<T>) {
    let mut guard = shared.inner.lock().unwrap();
    guard.stop = true;
    shared.condition.notify_all();
}
