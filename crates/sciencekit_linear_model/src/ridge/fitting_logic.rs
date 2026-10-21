//! Fit orchestration for `SKRidge`: representation dispatch, shape validation,
//! the `Fit` span, Gram accumulation and the dense/sparse solve
//! (spec `ridge-regression`).

use ndarray::{Array1, ArrayView2};
use sciencekit_common::{
    SKDataView, SKError, SKExecutionContext, SKExecutionPlan, SKFloat, SKOperationAttributes,
    SKOperationKind, execution::SKAccessPattern, sk_run_operation,
};
use sciencekit_math::SKBackendSelection;
use sciencekit_math::kernels::SK_AXIS_SUM_GRAIN;
use sprs::CsMatView;

use super::core_implementation::SKRidge;
use super::gram::SKCenteredGramState;
use super::model::SKRidgeModel;
use super::solving::{SKRidgeSolution, solve_ridge};
use super::sparse::solve_sparse_ridge;

/// Validate, open the `Fit` span and route the fit to the representation's path.
pub(super) fn fit_impl<F: SKBackendSelection>(
    estimator: &SKRidge<F>,
    data: SKDataView<'_, F>,
    target: Array1<F>,
) -> Result<SKRidgeModel<F>, SKError> {
    let (rows_count, columns_count) = match data {
        SKDataView::Dense(dense) => dense.dim(),
        SKDataView::Sparse(sparse) => (sparse.rows(), sparse.cols()),
        _ => return Err(unsupported_features()),
    };
    let (rows_count, columns_count) = validated_shape(rows_count, columns_count, target.len())?;
    let attributes = SKOperationAttributes {
        operation: SKOperationKind::Fit,
        rows_count,
        columns_count,
        execution_mode: estimator.execution_intent(),
        backend: estimator.backend,
    };
    sk_run_operation(attributes, || match data {
        SKDataView::Dense(dense) => fit_dense(estimator, dense, &target),
        SKDataView::Sparse(sparse) => fit_sparse(estimator, sparse, &target),
        _ => Err(unsupported_features()),
    })
}

/// The error for a feature representation the ridge estimators do not consume.
fn unsupported_features() -> SKError {
    SKError::UnsupportedRepresentation {
        representation: "unknown",
        suggestion: "convert the features to a dense matrix or a CSR matrix",
    }
}

/// Fit the dense path: accumulate the dense Gram and solve.
fn fit_dense<F: SKBackendSelection>(
    estimator: &SKRidge<F>,
    dense: ArrayView2<F>,
    target: &Array1<F>,
) -> Result<SKRidgeModel<F>, SKError> {
    let plan = resolve_dense_fit_plan(estimator, &dense)?;
    let mut gram = SKCenteredGramState::empty(dense.ncols());
    gram.accumulate_dense(&dense, &target.view(), plan.parallelism);
    let solution = solve_ridge(
        estimator.solver,
        "dense",
        estimator.include_intercept,
        &gram,
        dense,
        target.view(),
        &estimator.alphas,
        estimator.backend,
    )?;
    Ok(model_from(estimator, solution, gram))
}

/// Fit the sparse path: accumulate the sparse Gram and solve without
/// densifying the design.
fn fit_sparse<F: SKBackendSelection>(
    estimator: &SKRidge<F>,
    sparse: CsMatView<F>,
    target: &Array1<F>,
) -> Result<SKRidgeModel<F>, SKError> {
    let mut gram = SKCenteredGramState::empty(sparse.cols());
    gram.accumulate_sparse(&sparse, &target.view());
    let solution = solve_sparse_ridge(
        estimator.solver,
        &gram,
        sparse,
        target.view(),
        estimator.include_intercept,
        &estimator.alphas,
        estimator.tolerance,
        estimator.maximum_number_of_iterations,
        estimator.backend,
    )?;
    Ok(model_from(estimator, solution, gram))
}

/// Build the fitted model from a solve and the accumulated Gram state.
fn model_from<F: SKFloat>(
    estimator: &SKRidge<F>,
    solution: SKRidgeSolution<F>,
    gram: SKCenteredGramState<F>,
) -> SKRidgeModel<F> {
    SKRidgeModel::from_solve(
        estimator.alphas.clone(),
        solution.coefficients,
        solution.intercepts,
        solution.rank,
        solution.singular_values,
        solution.resolved,
        estimator.include_intercept,
        estimator.backend,
        estimator.execution_intent(),
        gram,
    )
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

/// Build a fresh execution context per operation, declare the reduction grain
/// and the `Sequential` access pattern, and resolve the plan.
fn resolve_dense_fit_plan<F: SKFloat>(
    estimator: &SKRidge<F>,
    features: &ArrayView2<F>,
) -> Result<SKExecutionPlan, SKError> {
    let mut context = SKExecutionContext::real();
    context.dataset_size_bytes = features.len() as u64 * size_of::<F>() as u64;
    context.scalar_size_bytes = size_of::<F>() as u64;
    context.dataset_elements = Some(features.len() as u64);
    context.access_pattern = SKAccessPattern::Sequential;
    context.grain = SK_AXIS_SUM_GRAIN;
    estimator.resolve_plan(&context)
}
