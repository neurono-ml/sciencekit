//! The driver's per-batch "partial fit": validation plus cross-product
//! accumulation (spec `linear-regression`).

use ndarray::ArrayView2;
use sciencekit_common::{
    SKError, SKExecutionContext, SKExecutionMode, SKFloat, SKOperationAttributes, SKOperationKind,
    SKSupervisedBatch, execution::SKAccessPattern, execution::SKStreamDecision,
    execution::sk_resolve_execution_plan, sk_run_operation,
};
use sciencekit_math::kernels::SK_AXIS_SUM_GRAIN;

use super::super::core_implementation::{SKLinearRegression, build_design};
use super::state::StreamingGramState;
use crate::linear_regression::gram::SKGramState;

/// Validate one supervised batch and merge its cross products into the running
/// state. The driver closure is infallible, so a shape violation is parked in
/// the state and halts the pipeline; the caller re-raises it.
pub(super) fn update_gram<F: SKFloat>(
    estimator: &SKLinearRegression,
    batch: SKSupervisedBatch<F>,
    state: &mut StreamingGramState<F>,
) -> SKStreamDecision {
    let features = batch.features();
    let targets = batch.targets();
    if state.columns_count == 0 {
        state.columns_count = features.ncols();
        state.gram =
            SKGramState::empty(state.columns_count + usize::from(estimator.include_intercept()));
    }
    if state.columns_count != features.ncols() {
        state.error = Some(SKError::ShapeMismatch {
            expected: vec![state.columns_count],
            found: vec![features.ncols()],
        });
        return SKStreamDecision::Stop;
    }
    if features.nrows() == 0 {
        return SKStreamDecision::Continue;
    }
    if targets.len() != features.nrows() {
        state.error = Some(SKError::ShapeMismatch {
            expected: vec![features.nrows()],
            found: vec![targets.len()],
        });
        return SKStreamDecision::Stop;
    }
    let attributes = SKOperationAttributes {
        operation: SKOperationKind::PartialFit,
        rows_count: features.nrows(),
        columns_count: features.ncols(),
        execution_mode: estimator.execution_intent(),
        backend: estimator.backend(),
    };
    sk_run_operation(attributes, || -> Result<(), SKError> {
        let parallelism =
            batch_parallelism(&features, state.available_memory_bytes, state.cpu_cores);
        let design = build_design(&features, estimator.include_intercept());
        state.gram.accumulate(&design.view(), &targets, parallelism);
        Ok(())
    })
    .expect("cross-product accumulation is infallible");
    SKStreamDecision::Continue
}

/// Resolve parallelism for one batch: an `Automatic` query over the batch as
/// the work unit, so large batches earn threads without any machine reads in
/// the decision path. Shared with the model's incremental `partial_fit`.
pub(crate) fn batch_parallelism<F: SKFloat>(
    data: &ArrayView2<F>,
    available_memory_bytes: u64,
    cpu_cores: usize,
) -> usize {
    let context = SKExecutionContext {
        available_memory_bytes,
        cpu_cores,
        dataset_size_bytes: data.len() as u64 * size_of::<F>() as u64,
        dataset_elements: Some(data.len() as u64),
        scalar_size_bytes: size_of::<F>() as u64,
        grain: SK_AXIS_SUM_GRAIN,
        access_pattern: SKAccessPattern::Sequential,
        batch_size_hint: None,
    };
    sk_resolve_execution_plan(SKExecutionMode::Automatic, &context)
        .map(|plan| plan.parallelism)
        .unwrap_or(1)
}
