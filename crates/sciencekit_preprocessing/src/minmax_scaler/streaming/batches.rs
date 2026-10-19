//! The driver's per-batch "partial fit": validation plus extrema merging.

use ndarray::{Array1, ArrayView2};
use sciencekit_common::{
    SKDataBatch, SKError, SKExecutionContext, SKExecutionMode, SKFloat, SKOperationAttributes,
    SKOperationKind,
    execution::{SKAccessPattern, SKStreamDecision, sk_resolve_execution_plan},
    observability::sk_run_operation,
};
use sciencekit_math::kernels::{SK_AXIS_SUM_GRAIN, sk_axis_min_max};

use super::super::core_implementation::SKMinMaxScaler;
use super::state::{StreamingExtrema, merge_batch};

/// Validate one batch and merge its extrema into the running state. The driver
/// closure is infallible, so a shape violation is parked in the state and halts
/// the pipeline; the caller re-raises it.
pub(super) fn update_extrema<F: SKFloat>(
    estimator: &SKMinMaxScaler<F>,
    batch: SKDataBatch<F>,
    state: &mut StreamingExtrema<F>,
) -> SKStreamDecision {
    let data = batch.data();
    if state.columns_count == 0 {
        state.columns_count = data.ncols();
        state.minimum = Array1::zeros(data.ncols());
        state.data_range = Array1::zeros(data.ncols());
    }
    if state.columns_count != data.ncols() {
        state.error = Some(SKError::ShapeMismatch {
            expected: vec![state.columns_count],
            found: vec![data.ncols()],
        });
        return SKStreamDecision::Stop;
    }
    if data.nrows() == 0 {
        return SKStreamDecision::Continue;
    }
    let attributes = SKOperationAttributes {
        operation: SKOperationKind::PartialFit,
        rows_count: data.nrows(),
        columns_count: data.ncols(),
        execution_mode: estimator.execution_intent(),
        backend: estimator.backend,
    };
    sk_run_operation(attributes, || -> Result<(), SKError> {
        let parallelism = batch_parallelism(&data, state.available_memory_bytes, state.cpu_cores);
        let (batch_minimum, batch_maximum) = batch_extrema(&data, parallelism);
        if state.count == 0 {
            // First batch: the running extrema are exactly this batch's.
            state.minimum.assign(&batch_minimum);
            state.data_range.assign(&(&batch_maximum - &batch_minimum));
            state.count = data.nrows() as u64;
        } else {
            merge_batch(state, &batch_minimum, &batch_maximum, data.nrows() as u64);
        }
        Ok(())
    })
    .expect("batch extrema update is infallible");
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

/// One batch's per-column minimum and maximum, over the reduction kernel.
/// Shared with `partial_fit`.
pub(crate) fn batch_extrema<F: SKFloat>(
    data: &ArrayView2<F>,
    parallelism: usize,
) -> (Array1<F>, Array1<F>) {
    sk_axis_min_max(data, 0, parallelism)
}
