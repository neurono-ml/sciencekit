//! The driver's per-batch sketch update: validation plus sketch feeding.

use sciencekit_common::{
    SKDataBatch, SKError, SKFloat, SKOperationAttributes, SKOperationKind,
    execution::SKStreamDecision, observability::sk_run_operation,
};

use super::super::core_implementation::SKRobustScaler;
use super::super::sketch::{extend_sketches, new_sketches};
use super::state::StreamingSketches;

/// Validate one batch and feed it into the running per-column sketches. The
/// driver closure is infallible, so a shape violation is parked in the state
/// and halts the pipeline; the caller re-raises it.
pub(super) fn update_sketches<F: SKFloat>(
    estimator: &SKRobustScaler<F>,
    batch: SKDataBatch<F>,
    state: &mut StreamingSketches,
) -> SKStreamDecision {
    let data = batch.data();
    if state.columns_count == 0 {
        state.columns_count = data.ncols();
        state.sketches = new_sketches(data.ncols());
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
        extend_sketches(&mut state.sketches, &data);
        state.count += data.nrows() as u64;
        Ok(())
    })
    .expect("batch sketch update is infallible");
    SKStreamDecision::Continue
}
