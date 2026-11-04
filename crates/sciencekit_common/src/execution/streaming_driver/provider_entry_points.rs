//! Provider-based streaming entry points: one source per pass (spec
//! `streaming-source-provider`).

use super::pass_policy::SKStreamingPassPolicy;
use super::prefetch::{SKStreamDecision, run_prefetch_driver};
use crate::SKError;
use crate::SKFloat;
use crate::batching::{
    SKDataBatch, SKLazySource, SKSourceProvider, SKSupervisedBatch, SKSupervisedLazySource,
    SKSupervisedSourceProvider,
};
use crate::execution::SKExecutionPlan;

/// Run the policy's passes over a feature-only provider, returning the passes
/// actually taken.
///
/// Each pass opens a fresh source through the provider and drives it through
/// the prefetch pipeline; block order is preserved within every pass. A
/// one-shot provider yields a single pass even when the policy asks for more,
/// and the clamp is reported with a warning. A factory or source failure aborts
/// the fit with the central taxonomy error.
pub fn sk_run_streaming_provider<F, Provider, State, Update>(
    provider: &mut Provider,
    state: &mut State,
    update: Update,
    plan: &SKExecutionPlan,
    policy: SKStreamingPassPolicy,
) -> Result<usize, SKError>
where
    F: SKFloat + Send,
    Provider: SKSourceProvider<F>,
    Update: Fn(SKDataBatch<F>, usize, &mut State) -> SKStreamDecision,
{
    let pass_count = sk_resolve_feature_pass_count::<F, Provider>(policy);
    let mut passes_taken = 0;
    for _pass in 0..pass_count {
        let mut source = provider.open()?;
        let decision = run_prefetch_driver(
            || source.next_batch().map_err(Into::into),
            state,
            &update,
            plan,
        )?;
        passes_taken += 1;
        if decision == SKStreamDecision::Stop {
            break;
        }
    }
    Ok(passes_taken)
}

/// Run the policy's passes over a supervised provider, returning the passes
/// actually taken.
///
/// The supervised twin of [`sk_run_streaming_provider`]: each block pairs its
/// features with its targets.
pub fn sk_run_supervised_streaming_provider<F, Provider, State, Update>(
    provider: &mut Provider,
    state: &mut State,
    update: Update,
    plan: &SKExecutionPlan,
    policy: SKStreamingPassPolicy,
) -> Result<usize, SKError>
where
    F: SKFloat + Send,
    Provider: SKSupervisedSourceProvider<F>,
    Update: Fn(SKSupervisedBatch<F>, usize, &mut State) -> SKStreamDecision,
{
    let pass_count = sk_resolve_supervised_pass_count::<F, Provider>(policy);
    let mut passes_taken = 0;
    for _pass in 0..pass_count {
        let mut source = provider.open()?;
        let decision = run_prefetch_driver(
            || source.next_supervised_batch().map_err(Into::into),
            state,
            &update,
            plan,
        )?;
        passes_taken += 1;
        if decision == SKStreamDecision::Stop {
            break;
        }
    }
    Ok(passes_taken)
}

fn sk_resolve_feature_pass_count<F, Provider>(policy: SKStreamingPassPolicy) -> usize
where
    Provider: SKSourceProvider<F>,
{
    sk_resolve_pass_count(policy, Provider::IS_REOPENABLE)
}

fn sk_resolve_supervised_pass_count<F, Provider>(policy: SKStreamingPassPolicy) -> usize
where
    Provider: SKSupervisedSourceProvider<F>,
{
    sk_resolve_pass_count(policy, Provider::IS_REOPENABLE)
}

fn sk_resolve_pass_count(policy: SKStreamingPassPolicy, is_reopenable: bool) -> usize {
    if let SKStreamingPassPolicy::Iterative {
        maximum_number_of_passes,
    } = policy
    {
        if !is_reopenable && maximum_number_of_passes > 1 {
            tracing::warn!(
                maximum_number_of_passes,
                "a one-shot source provider cannot reopen; taking a single streaming pass"
            );
        }
    }
    policy.resolve_pass_count(is_reopenable)
}
