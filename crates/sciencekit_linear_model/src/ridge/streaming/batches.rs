//! The driver's per-batch "partial fit": validation plus centered cross-product
//! accumulation (spec `ridge-regression`).

use ndarray::ArrayView2;
use sciencekit_common::{
    SKError, SKExecutionContext, SKExecutionMode, SKFloat, SKOperationAttributes, SKOperationKind,
    SKSupervisedBatch, execution::SKAccessPattern, execution::SKStreamDecision,
    execution::sk_resolve_execution_plan, sk_run_operation,
};
use sciencekit_math::kernels::SK_AXIS_SUM_GRAIN;

use super::state::StreamingCenteredGramState;
use crate::ridge::core_implementation::SKRidge;
use crate::ridge::gram::SKCenteredGramState;

/// Validate one supervised batch and merge its centered cross products into the
/// running state. The driver closure is infallible, so a shape violation is
/// parked in the state and halts the pipeline; the caller re-raises it.
pub(super) fn update_gram<F: SKFloat>(
    estimator: &SKRidge<F>,
    batch: SKSupervisedBatch<F>,
    state: &mut StreamingCenteredGramState<F>,
) -> SKStreamDecision {
    let features = batch.features();
    let targets = batch.targets();
    if state.columns_count == 0 {
        state.columns_count = features.ncols();
        state.gram = SKCenteredGramState::empty(state.columns_count);
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
        state
            .gram
            .accumulate_dense(&features, &targets, parallelism);
        Ok(())
    })
    .expect("cross-product accumulation is infallible");
    SKStreamDecision::Continue
}

/// Resolve parallelism for one batch: an `Automatic` query over the batch as the
/// work unit, so large batches earn threads without any machine reads in the
/// decision path. Shared with the model's incremental `partial_fit`.
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
