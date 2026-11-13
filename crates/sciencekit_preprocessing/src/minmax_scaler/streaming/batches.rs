//! The driver's per-batch "partial fit": validation plus extrema merging.

use ndarray::{Array1, ArrayView2};
use sciencekit_common::{
    SKDataBatch, SKDataView, SKError, SKExecutionContext, SKExecutionMode, SKFloat,
    SKOperationAttributes, SKOperationKind,
    execution::{SKAccessPattern, SKStreamDecision, sk_resolve_execution_plan},
    observability::sk_run_operation,
};
use sciencekit_math::kernels::{SK_AXIS_SUM_GRAIN, sk_axis_min_max};
use sprs::CsMatView;

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
        state.minimum = Array1::zeros(columns_count);
        state.maximum = Array1::zeros(columns_count);
        state.data_range = Array1::zeros(columns_count);
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
        let (batch_minimum, batch_maximum) = match view {
            SKDataView::Dense(data) => {
                let parallelism =
                    batch_parallelism(&data, state.available_memory_bytes, state.cpu_cores);
                batch_extrema(&data, parallelism)
            }
            SKDataView::Sparse(data) => sparse_batch_extrema(&data),
            _ => unreachable!("the shape check rejected every other representation"),
        };
        if state.count == 0 {
            // First batch: the running extrema are exactly this batch's.
            state.minimum.assign(&batch_minimum);
            state.maximum.assign(&batch_maximum);
            state.data_range.assign(&(&batch_maximum - &batch_minimum));
            state.count = rows_count as u64;
        } else {
            merge_batch(state, &batch_minimum, &batch_maximum, rows_count as u64);
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
        batch_byte_size_hint: None,
        device: None,
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

/// One sparse batch's per-column extrema in `O(nnz + p)`.
///
/// An implicit zero participates in a column's extrema whenever that column has
/// no explicit entry in some row; a column explicitly dense in every row keeps
/// its true extrema, so a sparse matrix of strictly positive values is never
/// lowered to zero artificially.
pub(crate) fn sparse_batch_extrema<F: SKFloat>(
    features: &CsMatView<'_, F>,
) -> (Array1<F>, Array1<F>) {
    let mut minimum = Array1::from_elem(features.cols(), F::infinity());
    let mut maximum = Array1::from_elem(features.cols(), F::neg_infinity());
    let mut explicit_counts = vec![0usize; features.cols()];
    for sparse_row in features.outer_iterator() {
        for (&column, &value) in sparse_row.indices().iter().zip(sparse_row.data()) {
            explicit_counts[column] += 1;
            minimum[column] = minimum[column].min(value);
            maximum[column] = maximum[column].max(value);
        }
    }
    let rows = features.rows();
    for column in 0..features.cols() {
        if explicit_counts[column] < rows {
            minimum[column] = minimum[column].min(F::zero());
            maximum[column] = maximum[column].max(F::zero());
        }
    }
    (minimum, maximum)
}
