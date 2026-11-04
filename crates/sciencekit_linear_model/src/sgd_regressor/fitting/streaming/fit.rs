//! The multi-epoch out-of-core regression fit (spec
//! `stochastic-gradient-regressor`, design decision 8).

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionContext, SKExecutionMode, SKExecutionPlan, SKFloat,
    SKOperationAttributes, SKOperationKind, SKSupervisedSourceProvider,
    execution::{
        SKAccessPattern, SKStreamingPassPolicy, sk_resolve_execution_plan,
        sk_run_supervised_streaming_provider,
    },
    sk_run_operation,
};
use sciencekit_math::kernels::SK_AXIS_SUM_GRAIN;

use super::super::super::coefficients::SKSGDRegressorCoefficients;
use super::super::super::model::SKSGDRegressorModel;
use super::super::state::sk_reported_parts;
use super::state::SKRegressorStreamState;
use crate::sgd::config::SKSGDConfig;
use crate::sgd::objective::SKEpochObjective;
use crate::sgd::{
    SKConvergenceTracker, SKRandomGenerator, SKSGDRegressorLoss, sk_finish_epochs,
    sk_objective_improved, sk_regressor_loss_adapter,
};

/// Fit on a source provider, running one epoch per pass.
///
/// A reopenable provider supplies the same budget as the in-memory fit, with
/// the learning-rate schedule counter and the averaged iterate continued
/// across passes. A one-shot provider supplies a single pass and never raises
/// for the missing budget.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sk_fit_streaming<F, Provider>(
    provider: Provider,
    config: &SKSGDConfig<F>,
    loss: SKSGDRegressorLoss,
    epsilon: F,
    warm_start: Option<&SKSGDRegressorCoefficients<F>>,
    backend: SKBackendKind,
    execution_intent: SKExecutionMode,
) -> Result<SKSGDRegressorModel<F>, SKError>
where
    F: SKFloat,
    Provider: SKSupervisedSourceProvider<F>,
{
    if config.early_stopping {
        return Err(SKError::UnsupportedOperation {
            operation: "early_stopping",
            reason: "a streaming source has no held-out slice to compare; use an in-memory fit"
                .to_string(),
        });
    }
    let attributes = SKOperationAttributes {
        operation: SKOperationKind::Fit,
        rows_count: 0,
        columns_count: 0,
        execution_mode: execution_intent,
        backend,
    };
    sk_run_operation(attributes, || {
        let plan = sk_resolve_streaming_plan()?;
        let mut provider = provider;
        let mut state = SKRegressorStreamState {
            config,
            loss,
            epsilon,
            warm_start,
            generator: SKRandomGenerator::new(config.random_seed),
            state: None,
            adapter: sk_regressor_loss_adapter(loss, epsilon),
            columns: 0,
            batches_this_pass: 0,
            objective: SKEpochObjective::new(),
            error: None,
        };
        let mut tracker = SKConvergenceTracker::new(
            config.tolerance,
            config.number_of_epochs_without_improvement,
        );
        let mut previous_objective: Option<F> = None;
        let mut epochs_taken = 0;
        let mut converged = false;
        for _ in 0..config.maximum_number_of_iterations {
            state.objective = SKEpochObjective::new();
            state.batches_this_pass = 0;
            epochs_taken += sk_run_supervised_streaming_provider(
                &mut provider,
                &mut state,
                |batch, _, state| state.update(batch),
                &plan,
                SKStreamingPassPolicy::Iterative {
                    maximum_number_of_passes: 1,
                },
            )?;
            if let Some(error) = state.error.take() {
                return Err(error);
            }
            if state.batches_this_pass == 0 {
                return Err(SKError::Conversion(
                    "streaming fit received no batches".into(),
                ));
            }
            let objective = state.objective.mean();
            let improved = previous_objective.is_none_or(|previous| {
                sk_objective_improved(previous, objective, config.tolerance)
            });
            previous_objective = Some(objective);
            if !improved {
                if let Some(live) = state.state.as_mut() {
                    live.halve_learning_rate();
                }
            }
            converged = tracker.record(objective);
            if converged || !Provider::IS_REOPENABLE {
                break;
            }
        }
        let provider_limited = !Provider::IS_REOPENABLE && config.maximum_number_of_iterations > 1;
        if provider_limited && config.maximum_number_of_iterations > 1 {
            tracing::warn!(
                maximum_number_of_iterations = config.maximum_number_of_iterations,
                "a one-shot source provider cannot reopen; taking a single streaming epoch"
            );
        }
        sk_finish_epochs(
            config.convergence_policy,
            config.maximum_number_of_iterations,
            epochs_taken,
            converged,
            provider_limited,
        )?;
        let live = state
            .state
            .take()
            .expect("a non-empty stream produced its state");
        let (coefficients, intercept) = sk_reported_parts(&live, state.columns);
        Ok(SKSGDRegressorModel::from_parts(
            coefficients,
            intercept,
            live,
            config,
            loss,
            epsilon,
            backend,
            execution_intent,
            converged,
            epochs_taken,
        ))
    })
}

/// Resolve the plan for an unbounded stream (Sequential access pattern).
fn sk_resolve_streaming_plan() -> Result<SKExecutionPlan, SKError> {
    let mut context = SKExecutionContext::real();
    context.dataset_size_bytes = u64::MAX;
    context.scalar_size_bytes = 8;
    context.dataset_elements = None;
    context.access_pattern = SKAccessPattern::Sequential;
    context.grain = SK_AXIS_SUM_GRAIN;
    sk_resolve_execution_plan(SKExecutionMode::Automatic, &context)
}
