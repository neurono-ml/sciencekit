//! The `SKRidge` estimator and its supervised dense `fit`
//! (spec `ridge-regression`).

use ndarray::ArrayView2;
use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKDataView, SKError, SKExecutionContext, SKExecutionMode,
    SKExecutionPlan, SKFloat, SKOperationAttributes, SKOperationKind, SKSupervisedFit,
    SKTargetView, execution::SKAccessPattern, sk_run_operation,
};
use sciencekit_math::SKBackendSelection;
use sciencekit_math::kernels::SK_AXIS_SUM_GRAIN;

use crate::ridge::builder::{SKRidgeBuilder, SKRidgeSolver};
use crate::ridge::gram::SKCenteredGramState;
use crate::ridge::model::SKRidgeModel;
use crate::ridge::solving::{reject_sparse, solve_ridge};

/// Ridge regression, builder-configured.
///
/// Immutable configuration carrying scalar values (`alpha`, `tolerance`), so —
/// unlike the scalar-free [`crate::linear_regression::SKLinearRegression`] — the
/// estimator is generic over the model scalar `F`. The learned state lives only
/// on the distinct [`SKRidgeModel`], so "not yet fitted" is unrepresentable and
/// the model is the export boundary.
#[derive(Debug, Clone)]
pub struct SKRidge<F> {
    pub(super) plan: SKBuilderState,
    pub(super) alphas: Vec<F>,
    pub(super) include_intercept: bool,
    pub(super) solver: SKRidgeSolver,
    pub(super) tolerance: F,
    pub(super) maximum_number_of_iterations: usize,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKRidge<F> {
    /// Start building an estimator (the only construction surface).
    pub fn build() -> SKRidgeBuilder<F> {
        SKRidgeBuilder::new()
    }

    /// The configured execution intent.
    pub fn execution_intent(&self) -> SKExecutionMode {
        self.plan.execution_intent()
    }

    /// Resolve the execution intent against a context.
    pub fn resolve_plan(&self, context: &SKExecutionContext) -> Result<SKExecutionPlan, SKError> {
        self.plan.resolve_plan(context)
    }

    /// The backend kind recorded in operation spans and transferred to models.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// Whether the fit solves for an unpenalized intercept.
    pub fn include_intercept(&self) -> bool {
        self.include_intercept
    }

    /// The configured solver portfolio selection.
    pub fn solver(&self) -> SKRidgeSolver {
        self.solver
    }

    /// The configured penalty values, in configuration order.
    pub fn alphas(&self) -> &[F] {
        &self.alphas
    }

    /// The conjugate-gradient residual tolerance.
    pub fn tolerance(&self) -> F {
        self.tolerance
    }

    /// The conjugate-gradient iteration budget.
    pub fn maximum_number_of_iterations(&self) -> usize {
        self.maximum_number_of_iterations
    }
}

impl<F: SKBackendSelection> SKSupervisedFit<F> for SKRidge<F> {
    type Model = SKRidgeModel<F>;
    type Error = SKError;

    fn fit<'a, X, T>(&self, features: X, targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        let data = features.try_into()?;
        let representation = data.representation();
        let dense = match data {
            SKDataView::Dense(dense) => dense,
            _ => return Err(reject_sparse(self.solver)),
        };
        let target = targets.try_into()?.as_continuous()?.into_owned();
        let (rows_count, columns_count) = validated_shape(&dense, target.len())?;
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Fit,
            rows_count,
            columns_count,
            execution_mode: self.execution_intent(),
            backend: self.backend,
        };
        sk_run_operation(attributes, || {
            let plan = resolve_fit_plan(self, &dense)?;
            let mut gram = SKCenteredGramState::empty(columns_count);
            gram.accumulate_dense(&dense, &target.view(), plan.parallelism);
            let solution = solve_ridge(
                self.solver,
                representation,
                self.include_intercept,
                &gram,
                dense.view(),
                target.view(),
                &self.alphas,
                self.backend,
            )?;
            Ok(SKRidgeModel::from_solve(
                self.alphas.clone(),
                solution.coefficients,
                solution.intercepts,
                solution.rank,
                solution.singular_values,
                solution.resolved,
                self.include_intercept,
                self.backend,
                self.execution_intent(),
                gram,
            ))
        })
    }
}

/// Reject empty inputs and a target count that disagrees with the row count,
/// before any element is processed.
fn validated_shape<F>(
    features: &ArrayView2<F>,
    target_len: usize,
) -> Result<(usize, usize), SKError> {
    let (rows_count, columns_count) = features.dim();
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
fn resolve_fit_plan<F: SKFloat>(
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
