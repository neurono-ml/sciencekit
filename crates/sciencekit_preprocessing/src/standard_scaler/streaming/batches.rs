//! The driver's per-batch "partial fit": validation plus moment merging.

use ndarray::{Array1, ArrayView2};
use sciencekit_common::{
    SKDataBatch, SKError, SKExecutionContext, SKExecutionMode, SKFloat, SKOperationAttributes,
    SKOperationKind,
    execution::{SKAccessPattern, SKStreamDecision, sk_resolve_execution_plan},
    observability::sk_run_operation,
};
use sciencekit_math::kernels::{SK_AXIS_SUM_GRAIN, sk_axis_sum, sk_axis_sum_squares};

use super::super::core_implementation::SKStandardScaler;
use super::state::{StreamingMoments, merge_batch};

/// Validate one batch and merge its moments into the running state. The driver
/// closure is infallible, so a shape violation is parked in the state and
/// halts the pipeline; the caller re-raises it.
pub(super) fn update_moments<F: SKFloat>(
    estimator: &SKStandardScaler,
    batch: SKDataBatch<F>,
    state: &mut StreamingMoments<F>,
) -> SKStreamDecision {
    let data = batch.data();
    if state.columns_count == 0 {
        state.columns_count = data.ncols();
        state.mean = Array1::zeros(data.ncols());
        state.sum_squared_deviations = Array1::zeros(data.ncols());
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
        rows: data.nrows(),
        columns: data.ncols(),
        execution_mode: estimator.execution_intent(),
        backend: estimator.backend,
    };
    sk_run_operation(attributes, || -> Result<(), SKError> {
        let parallelism = batch_parallelism(&data, state.available_memory_bytes, state.cpu_cores);
        let (batch_mean, batch_deviations) = batch_moments(&data, parallelism);
        merge_batch(state, &batch_mean, &batch_deviations, data.nrows() as u64);
        Ok(())
    })
    .expect("batch moment update is infallible");
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

/// One batch's mean and Σ(x − μ)², the latter via the exact identity
/// Σx² − 2μΣx + nμ² over the reduction kernels. Shared with `partial_fit`.
pub(crate) fn batch_moments<F: SKFloat>(
    data: &ArrayView2<F>,
    parallelism: usize,
) -> (Array1<F>, Array1<F>) {
    let rows_as_scalar = F::from(data.nrows()).expect("batch row count fits in scalar");
    let sums = sk_axis_sum(data, 0, parallelism);
    let sum_squares = sk_axis_sum_squares(data, 0, parallelism);
    let mut mean = Array1::zeros(data.ncols());
    let mut deviations = Array1::zeros(data.ncols());
    for (index, (sum, sum_squares)) in sums.iter().zip(sum_squares.iter()).enumerate() {
        let column_mean = *sum / rows_as_scalar;
        mean[index] = column_mean;
        deviations[index] = *sum_squares - F::from(2).unwrap() * column_mean * *sum
            + rows_as_scalar * column_mean * column_mean;
    }
    (mean, deviations)
}
