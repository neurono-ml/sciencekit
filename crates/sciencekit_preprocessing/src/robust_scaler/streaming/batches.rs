//! The driver's per-batch sketch update: validation plus sketch feeding.

use sciencekit_common::{
    SKDataBatch, SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind,
    execution::SKStreamDecision, observability::sk_run_operation,
};

use super::super::core_implementation::SKRobustScaler;
use super::super::sketch::{extend_sketches, extend_sparse_sketches, new_sketches};
use super::state::StreamingSketches;

/// Validate one batch and feed it into the running per-column sketches. The
/// driver closure is infallible, so a shape violation is parked in the state
/// and halts the pipeline; the caller re-raises it.
pub(super) fn update_sketches<F: SKFloat>(
    estimator: &SKRobustScaler<F>,
    batch: SKDataBatch<F>,
    state: &mut StreamingSketches,
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
        state.sketches = new_sketches(columns_count);
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
        match view {
            SKDataView::Dense(data) => extend_sketches(&mut state.sketches, &data),
            SKDataView::Sparse(data) => extend_sparse_sketches(&mut state.sketches, &data),
            _ => unreachable!("the shape check rejected every other representation"),
        }
        state.count += rows_count as u64;
        Ok(())
    })
    .expect("batch sketch update is infallible");
    SKStreamDecision::Continue
}
