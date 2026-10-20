//! The two public streaming drivers, one per source kind.

use super::prefetch::{SKStreamDecision, run_prefetch_driver};
use crate::SKError;
use crate::SKFloat;
use crate::batching::{SKDataBatch, SKLazySource, SKSupervisedBatch, SKSupervisedLazySource};
use crate::execution::SKExecutionPlan;

/// Run the prefetch pipeline over a feature-only [`SKLazySource`].
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
    run_prefetch_driver(
        || source.next_batch().map_err(Into::into),
        state,
        update,
        plan,
    )
}

/// Run the prefetch pipeline over a supervised [`SKSupervisedLazySource`].
pub fn sk_run_supervised_streaming_driver<F, Source, State, Update>(
    source: &mut Source,
    state: &mut State,
    update: Update,
    plan: &SKExecutionPlan,
) -> Result<(), SKError>
where
    F: SKFloat + Send,
    Source: SKSupervisedLazySource<F> + Send,
    Source::Error: From<SKError> + Into<SKError>,
    Update: Fn(SKSupervisedBatch<F>, usize, &mut State) -> SKStreamDecision,
{
    run_prefetch_driver(
        || source.next_supervised_batch().map_err(Into::into),
        state,
        update,
        plan,
    )
}
