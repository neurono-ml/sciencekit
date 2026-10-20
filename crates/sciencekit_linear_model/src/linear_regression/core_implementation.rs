//! The `SKLinearRegression` estimator and its supervised dense `fit`
//! (spec `linear-regression`).

use ndarray::{Array2, ArrayView2, s};
use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKDataView, SKError, SKExecutionContext, SKExecutionMode,
    SKExecutionPlan, SKFloat, SKOperationAttributes, SKOperationKind, SKSupervisedFit,
    SKTargetView, execution::SKAccessPattern, sk_run_operation,
};
use sciencekit_math::kernels::SK_AXIS_SUM_GRAIN;
use sciencekit_math::{SKFaerBackend, SKMathBackend, SKMatrixMultiplyBackend};

use crate::linear_regression::builder::SKLinearRegressionBuilder;
use crate::linear_regression::gram::SKGramState;
use crate::linear_regression::model::SKLinearRegressionModel;
use crate::linear_regression::solving::solve_least_squares;

/// Ordinary least squares, builder-configured.
///
/// Immutable configuration with no scalar field — the estimator is **not**
/// generic over the scalar; the fitted [`SKLinearRegressionModel`] is. The
/// learned state lives only on the model, so "not yet fitted" is
/// unrepresentable and the model is the export boundary.
#[derive(Debug, Clone, Copy)]
pub struct SKLinearRegression {
    pub(super) plan: SKBuilderState,
    pub(super) include_intercept: bool,
    pub(super) backend: SKBackendKind,
}

impl SKLinearRegression {
    /// Start building an estimator (the only construction surface).
    pub fn build() -> SKLinearRegressionBuilder {
        SKLinearRegressionBuilder::new()
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

    /// Whether the fit solves for an intercept.
    pub fn include_intercept(&self) -> bool {
        self.include_intercept
    }
}

impl<F: SKFloat> SKSupervisedFit<F> for SKLinearRegression
where
    SKFaerBackend: SKMathBackend<F>,
    SKMatrixMultiplyBackend: SKMathBackend<F>,
{
    type Model = SKLinearRegressionModel<F>;
    type Error = SKError;

    fn fit<'a, X, T>(&self, features: X, targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        let dense = features.try_into()?.as_dense()?;
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
            let design = build_design(&dense, self.include_intercept);
            let gram = gram_of(&design, &target, plan.parallelism);
            let solution = solve_least_squares(self.backend, design.view(), target.view())?;
            Ok(SKLinearRegressionModel::from_solution(
                solution,
                gram,
                self.include_intercept,
                self.backend,
                self.execution_intent(),
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

/// The design matrix: `[1 ‖ X]` when the intercept is included, else `X`.
pub(super) fn build_design<F: SKFloat>(
    features: &ArrayView2<F>,
    include_intercept: bool,
) -> Array2<F> {
    let (rows_count, columns_count) = features.dim();
    if !include_intercept {
        return features.to_owned();
    }
    let mut design = Array2::zeros((rows_count, columns_count + 1));
    design.column_mut(0).fill(F::one());
    design.slice_mut(s![.., 1..]).assign(features);
    design
}

/// The Gram state of a design matrix and its targets (for incremental
/// continuation after a dense fit).
fn gram_of<F: SKFloat>(
    design: &Array2<F>,
    target: &ndarray::Array1<F>,
    parallelism: usize,
) -> SKGramState<F> {
    let mut gram = SKGramState::empty(design.ncols());
    gram.accumulate(&design.view(), &target.view(), parallelism);
    gram
}

/// Build a fresh execution context per operation, declare the reduction grain
/// and the `Sequential` access pattern, and resolve the plan.
fn resolve_fit_plan<F: SKFloat>(
    estimator: &SKLinearRegression,
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
