//! The one-pass streaming drivers kept as thin provider adapters.

use super::pass_policy::SKStreamingPassPolicy;
use super::provider_entry_points::{
    sk_run_streaming_provider, sk_run_supervised_streaming_provider,
};
use crate::SKError;
use crate::SKFloat;
use crate::batching::{SKDataBatch, SKLazySource, SKSupervisedBatch, SKSupervisedLazySource};
use crate::execution::SKExecutionPlan;

/// Run the prefetch pipeline once over a feature-only [`SKLazySource`].
///
/// A thin adapter over [`sk_run_streaming_provider`] using the residual
/// one-shot provider, so existing call sites keep compiling unchanged.
pub fn sk_run_streaming_driver<F, Source, State, Update>(
    source: &mut Source,
    state: &mut State,
    update: Update,
    plan: &SKExecutionPlan,
) -> Result<(), SKError>
where
    F: SKFloat + Send,
    Source: SKLazySource<F> + Send,
    // Only `Into` is required by the provider residual implementation; the
    // redundant `From` was kept for compatibility with callers whose concrete
    // source errors are built by converting an `SKError` with `?`.
    Source::Error: From<SKError> + Into<SKError>,
    Update: Fn(SKDataBatch<F>, usize, &mut State) -> super::SKStreamDecision,
{
    let mut provider = source;
    sk_run_streaming_provider::<F, _, _, _>(
        &mut provider,
        state,
        update,
        plan,
        SKStreamingPassPolicy::SinglePass,
    )?;
    Ok(())
}

/// Run the prefetch pipeline once over a supervised [`SKSupervisedLazySource`].
///
/// A thin adapter over [`sk_run_supervised_streaming_provider`] using the
/// residual one-shot provider, so existing call sites keep compiling unchanged.
pub fn sk_run_supervised_streaming_driver<F, Source, State, Update>(
    source: &mut Source,
    state: &mut State,
    update: Update,
    plan: &SKExecutionPlan,
) -> Result<(), SKError>
where
    F: SKFloat + Send,
    Source: SKSupervisedLazySource<F> + Send,
    // Only `Into` is required by the provider residual implementation; the
    // redundant `From` was kept for compatibility with callers whose concrete
    // source errors are built by converting an `SKError` with `?`.
    Source::Error: From<SKError> + Into<SKError>,
    Update: Fn(SKSupervisedBatch<F>, usize, &mut State) -> super::SKStreamDecision,
{
    let mut provider = source;
    sk_run_supervised_streaming_provider::<F, _, _, _>(
        &mut provider,
        state,
        update,
        plan,
        SKStreamingPassPolicy::SinglePass,
    )?;
    Ok(())
}
