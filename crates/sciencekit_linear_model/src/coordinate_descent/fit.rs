//! Fit orchestration shared by `SKLasso` and `SKElasticNet`: shape validation,
//! representation dispatch, the `Fit` span, design accumulation and the solve
//! (spec `coordinate-descent-solver`).

use ndarray::Array1;
use sciencekit_common::{
    SKBackendKind, SKDataView, SKError, SKExecutionMode, SKFloat, SKOperationAttributes,
    SKOperationKind, sk_run_operation,
};

use super::accumulator::SKCoordinateDescentState;
use super::config::SKCoordinateDescentConfig;
use super::core::SKCoordinateDescentModelCore;

/// Validate, open the `Fit` span, accumulate the design and solve.
#[allow(clippy::too_many_arguments)]
pub(crate) fn sk_fit_coordinate_descent<F: SKFloat>(
    data: SKDataView<'_, F>,
    target: Array1<F>,
    alphas: &[F],
    l1_ratio: F,
    include_intercept: bool,
    backend: SKBackendKind,
    execution_intent: SKExecutionMode,
    config: SKCoordinateDescentConfig<F>,
) -> Result<SKCoordinateDescentModelCore<F>, SKError> {
    let (rows_count, columns_count) = match data {
        SKDataView::Dense(dense) => dense.dim(),
        SKDataView::Sparse(sparse) => (sparse.rows(), sparse.cols()),
        _ => return Err(unsupported_representation()),
    };
    let (rows_count, columns_count) = validated_shape(rows_count, columns_count, target.len())?;
    let attributes = SKOperationAttributes {
        operation: SKOperationKind::Fit,
        rows_count,
        columns_count,
        execution_mode: execution_intent,
        backend,
    };
    sk_run_operation(attributes, || {
        let mut state = match data {
            SKDataView::Dense(_) => SKCoordinateDescentState::new_dense(),
            SKDataView::Sparse(_) => SKCoordinateDescentState::new_sparse(),
            _ => return Err(unsupported_representation()),
        };
        match data {
            SKDataView::Dense(dense) => state.accumulate_dense(dense, target.view()),
            SKDataView::Sparse(sparse) => state.accumulate_sparse(sparse, target.view()),
            _ => return Err(unsupported_representation()),
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

/// The error for a feature representation the estimators do not consume.
fn unsupported_representation() -> SKError {
    SKError::UnsupportedRepresentation {
        representation: "unknown",
        suggestion: "convert the features to a dense matrix or a CSR matrix",
    }
}

/// Reject empty inputs and a target count that disagrees with the row count,
/// before any element is processed.
fn validated_shape(
    rows_count: usize,
    columns_count: usize,
    target_len: usize,
) -> Result<(usize, usize), SKError> {
    if rows_count == 0 || columns_count == 0 {
        return Err(SKError::ShapeMismatch {
            expected: vec![rows_count.max(1), columns_count.max(1)],
            found: vec![rows_count, columns_count],
        });
    }
    if target_len != rows_count {
        return Err(SKError::ShapeMismatch {
            expected: vec![rows_count],
            found: vec![target_len],
        });
    }
    Ok((rows_count, columns_count))
}
