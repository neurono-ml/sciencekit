//! The shared prefetch pipeline engine and the decision it is steered by.

use std::sync::Arc;

use super::channel::{Item, Shared, next_item, request_stop, run_producer};
use crate::SKError;
use crate::execution::SKExecutionPlan;

/// The decision an update step returns to steer the pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKStreamDecision {
    /// Keep reading and computing the next batch.
    Continue,
    /// Stop the pipeline: keep the current batch's effect, read no further.
    Stop,
}

/// Run the prefetch pipeline over a `read` closure, applying `update` to each
/// owned batch in source order.
///
/// A dedicated I/O thread reads ahead into a bounded buffer of the plan's
/// [`SKExecutionPlan::buffer_depth`]; the compute side runs `update` on the
/// caller's thread, receiving the batch, the plan's `parallelism` and the
/// mutable algorithm state. `update` returns [`SKStreamDecision::Continue`] to
/// keep going or [`SKStreamDecision::Stop`] to halt the pipeline, keeping the
/// current batch's effect. A mid-stream read failure surfaces as the central
/// taxonomy error with the effects of earlier batches preserved.
pub(super) fn run_prefetch_driver<T, State, Read, Update>(
    read: Read,
    state: &mut State,
    update: Update,
    plan: &SKExecutionPlan,
) -> Result<(), SKError>
where
    T: Send,
    Read: FnMut() -> Result<Option<T>, SKError> + Send,
    Update: Fn(T, usize, &mut State) -> SKStreamDecision,
{
    let shared = Arc::new(Shared::new(plan.buffer_depth));
    std::thread::scope(|scope| {
        let producer_shared = Arc::clone(&shared);
        let handle = scope.spawn(move || run_producer(read, &producer_shared));

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
