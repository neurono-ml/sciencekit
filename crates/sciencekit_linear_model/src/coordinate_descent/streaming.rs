//! Out-of-core fit orchestration shared by `SKLasso` and `SKElasticNet`
//! (spec `coordinate-descent-solver`, Decision 7).

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKFloat, SKOperationAttributes, SKOperationKind,
    SKSupervisedLazySource, sk_run_operation,
};

use super::accumulator::SKCoordinateDescentState;
use super::config::SKCoordinateDescentConfig;
use super::core::SKCoordinateDescentModelCore;

/// Fit on a sequential supervised source, retaining each batch's design and
/// solving the whole accumulated design once at the end.
///
/// The result equals the dense `fit` on the concatenated data within floating
/// point tolerance, regardless of batch order or size. An empty source is
/// rejected with a structured error.
pub(crate) fn sk_fit_streaming<F, S>(
    source: &mut S,
    alphas: &[F],
    l1_ratio: F,
    include_intercept: bool,
    backend: SKBackendKind,
    execution_intent: SKExecutionMode,
    config: SKCoordinateDescentConfig<F>,
) -> Result<SKCoordinateDescentModelCore<F>, SKError>
where
    F: SKFloat,
    S: SKSupervisedLazySource<F> + Send,
    S::Error: From<SKError> + Into<SKError>,
{
    // Single-pass source: totals are unknown upfront, so the outer span carries
    // zero dimensions.
    let attributes = SKOperationAttributes {
        operation: SKOperationKind::Fit,
        rows_count: 0,
        columns_count: 0,
        execution_mode: execution_intent,
        backend,
    };
    sk_run_operation(attributes, || {
        let state = collect_batches(source)?;
        if state.is_empty() {
            return Err(SKError::Conversion(
                "streaming fit received no batches".into(),
            ));
        }
        let solution = state.solve(alphas, l1_ratio, include_intercept, &config)?;
        Ok(SKCoordinateDescentModelCore::from_solution(
            alphas.to_vec(),
            l1_ratio,
            solution,
            include_intercept,
            backend,
            execution_intent,
            config,
            state,
        ))
    })
}

/// Retain every batch's design and targets into one accumulator, validating the
/// column count.
fn collect_batches<F, S>(source: &mut S) -> Result<SKCoordinateDescentState<F>, SKError>
where
    F: SKFloat,
    S: SKSupervisedLazySource<F> + Send,
    S::Error: From<SKError> + Into<SKError>,
{
    let mut state = SKCoordinateDescentState::new_dense();
    while let Some(batch) = source.next_supervised_batch().map_err(Into::into)? {
        let features = batch.features();
        let targets = batch.targets();
        if features.nrows() == 0 {
            continue;
        }
        if targets.len() != features.nrows() {
            return Err(SKError::ShapeMismatch {
                expected: vec![features.nrows()],
                found: vec![targets.len()],
            });
        }
        let columns_count = state.columns_count();
        if columns_count != 0 && columns_count != features.ncols() {
            return Err(SKError::ShapeMismatch {
                expected: vec![columns_count],
                found: vec![features.ncols()],
            });
        }
        state.accumulate_dense(features, targets);
    }
    Ok(state)
}
