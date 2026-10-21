//! The `SKRidge` estimator and its supervised dense `fit`
//! (spec `ridge-regression`).

use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKDataView, SKError, SKExecutionContext, SKExecutionMode,
    SKExecutionPlan, SKFloat, SKSupervisedFit, SKTargetView,
};
use sciencekit_math::SKBackendSelection;

use crate::ridge::builder::{SKRidgeBuilder, SKRidgeSolver};
use crate::ridge::fitting_logic::fit_impl;
use crate::ridge::model::SKRidgeModel;

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
        let target = targets.try_into()?.as_continuous()?.into_owned();
        fit_impl(self, data, target)
    }
}
