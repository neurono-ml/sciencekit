//! The driver's per-batch update: validation plus state accumulation
//! (spec `simple-imputer`).

use sciencekit_common::{
    SKDataBatch, SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind,
    execution::SKStreamDecision, observability::sk_run_operation,
};

use super::super::core_implementation::SKSimpleImputer;
use super::super::state::SKImputationState;
use super::state::StreamingImputation;

/// Validate one batch and feed it into the running per-column state. The
/// driver closure is infallible, so a shape violation is parked in the state
/// and halts the pipeline; the caller re-raises it.
pub(super) fn update_imputation<F: SKFloat>(
    estimator: &SKSimpleImputer<F>,
    batch: SKDataBatch<F>,
    state: &mut StreamingImputation<F>,
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
        state.imputation = Some(SKImputationState::new(columns_count, estimator.strategy));
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
        if let Some(imputation) = state.imputation.as_mut() {
            match view {
                SKDataView::Dense(data) => {
                    imputation.extend_dense(&data, estimator.strategy, estimator.missing_values);
                }
                SKDataView::Sparse(data) => {
                    imputation.extend_sparse(&data, estimator.strategy, estimator.missing_values);
                }
                _ => unreachable!("the shape check rejected every other representation"),
            }
        }
        state.rows_count += rows_count as u64;
        Ok(())
    })
    .expect("batch state update is infallible");
    SKStreamDecision::Continue
}
