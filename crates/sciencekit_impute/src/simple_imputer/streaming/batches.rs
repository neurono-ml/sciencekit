//! The driver's per-batch update: validation plus state accumulation
//! (spec `simple-imputer`).

use sciencekit_common::{
    SKDataBatch, SKError, SKFloat, SKOperationAttributes, SKOperationKind,
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
    let data = batch.data();
    if state.columns_count == 0 {
        state.columns_count = data.ncols();
        state.imputation = Some(SKImputationState::new(data.ncols(), estimator.strategy));
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
        if let Some(imputation) = state.imputation.as_mut() {
            imputation.extend_dense(&data, estimator.strategy, estimator.missing_values);
        }
        state.rows_count += data.nrows() as u64;
        Ok(())
    })
    .expect("batch state update is infallible");
    SKStreamDecision::Continue
}
