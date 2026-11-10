//! The driver's per-batch "partial fit": validation plus moment merging.

use ndarray::{Array1, ArrayView2};
use sciencekit_common::{
    SKDataBatch, SKDataView, SKError, SKExecutionContext, SKExecutionMode, SKFloat,
    SKOperationAttributes, SKOperationKind,
    execution::{SKAccessPattern, SKStreamDecision, sk_resolve_execution_plan},
    observability::sk_run_operation,
};
use sciencekit_math::kernels::{SK_AXIS_SUM_GRAIN, sk_axis_sum, sk_axis_sum_squares};
use sprs::CsMatView;

use super::super::core_implementation::{SKStandardScaler, moments_from_sums};
use super::state::{StreamingMoments, merge_batch};

/// Validate one batch and merge its moments into the running state. The driver
/// closure is infallible, so a shape violation is parked in the state and
/// halts the pipeline; the caller re-raises it.
pub(super) fn update_moments<F: SKFloat>(
    estimator: &SKStandardScaler,
    batch: SKDataBatch<F>,
    state: &mut StreamingMoments<F>,
) -> SKStreamDecision {
    let view = batch.view();
    let (rows_count, columns_count) = match view {
        SKDataView::Dense(data) => data.dim(),
        SKDataView::Sparse(data) => (data.rows(), data.cols()),
        _ => {
            state.error = Some(SKError::UnsupportedRepresentation {
                representation: "unknown",
                suggestion: "convert the features to a dense matrix or a CSR matrix",
            });
            return SKStreamDecision::Stop;
        }
    };
    if state.columns_count == 0 {
        state.columns_count = columns_count;
        state.mean = Array1::zeros(columns_count);
        state.sum_squared_deviations = Array1::zeros(columns_count);
    }
    if state.columns_count != columns_count {
        state.error = Some(SKError::ShapeMismatch {
            expected: vec![state.columns_count],
            found: vec![columns_count],
        });
        return SKStreamDecision::Stop;
    }
    if rows_count == 0 {
        return SKStreamDecision::Continue;
    }
    let attributes = SKOperationAttributes {
        operation: SKOperationKind::PartialFit,
        rows_count,
        columns_count,
        execution_mode: estimator.execution_intent(),
        backend: estimator.backend,
    };
    sk_run_operation(attributes, || -> Result<(), SKError> {
        let (batch_mean, batch_deviations) = match view {
            SKDataView::Dense(data) => {
                let parallelism =
                    batch_parallelism(&data, state.available_memory_bytes, state.cpu_cores);
                batch_moments(&data, parallelism)
            }
            SKDataView::Sparse(data) => sparse_batch_moments(&data),
            _ => unreachable!("the shape check rejected every other representation"),
        };
        merge_batch(state, &batch_mean, &batch_deviations, rows_count as u64);
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
        batch_byte_size_hint: None,
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
    let sums = sk_axis_sum(data, 0, parallelism);
    let sum_squares = sk_axis_sum_squares(data, 0, parallelism);
    moments_from_sums(&sums, &sum_squares, data.nrows())
}

/// One sparse batch's mean and Σ(x − μ)² in `O(nnz)`.
///
/// Implicit zeros contribute nothing to `Σx` or `Σx²`, but they do count as
/// rows in the moments identity, so every column's mean and M2 include them.
pub(crate) fn sparse_batch_moments<F: SKFloat>(
    features: &CsMatView<'_, F>,
) -> (Array1<F>, Array1<F>) {
    let mut sums = Array1::zeros(features.cols());
    let mut sum_squares = Array1::zeros(features.cols());
    for sparse_row in features.outer_iterator() {
        for (&column, &value) in sparse_row.indices().iter().zip(sparse_row.data()) {
            sums[column] = sums[column] + value;
            sum_squares[column] = sum_squares[column] + value * value;
        }
    }
    moments_from_sums(&sums, &sum_squares, features.rows())
}
