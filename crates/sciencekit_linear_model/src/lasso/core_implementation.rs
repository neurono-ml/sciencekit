//! The `SKLasso` estimator and its supervised fit (spec `lasso`).

use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKError, SKExecutionContext, SKExecutionMode, SKExecutionPlan,
    SKFloat, SKSupervisedFit, SKTargetView,
};

use crate::coordinate_descent::SKCoordinateDescentConfig;
use crate::coordinate_descent::SKSelection;
use crate::coordinate_descent::sk_fit_coordinate_descent;
use crate::lasso::builder::{SKLassoBuilder, coordinate_descent_config};
use crate::lasso::model::SKLassoModel;

/// Lasso regression, builder-configured.
///
/// Immutable configuration carrying scalar values (`alpha`, `tolerance`), so the
/// estimator is generic over the model scalar `F`. The learned state lives only
/// on the distinct [`SKLassoModel`].
#[derive(Debug, Clone)]
pub struct SKLasso<F> {
    pub(super) plan: SKBuilderState,
    pub(super) alphas: Vec<F>,
    pub(super) include_intercept: bool,
    pub(super) positive: bool,
    pub(super) selection: SKSelection,
    pub(super) screening: bool,
    pub(super) tolerance: F,
    pub(super) maximum_number_of_iterations: usize,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKLasso<F> {
    /// Start building an estimator (the only construction surface).
    pub fn build() -> SKLassoBuilder<F> {
        SKLassoBuilder::new()
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

    /// Whether every coefficient is constrained to be non-negative.
    pub fn positive(&self) -> bool {
        self.positive
    }

    /// The coordinate sweep order.
    pub fn selection(&self) -> SKSelection {
        self.selection
    }

    /// Whether gap-safe screening is enabled.
    pub fn screening(&self) -> bool {
        self.screening
    }

    /// The configured penalty values, in configuration order.
    pub fn alphas(&self) -> &[F] {
        &self.alphas
    }

    /// The convergence tolerance.
    pub fn tolerance(&self) -> F {
        self.tolerance
    }

    /// The iteration budget.
    pub fn maximum_number_of_iterations(&self) -> usize {
        self.maximum_number_of_iterations
    }

    /// The coordinate-descent configuration this estimator resolves to.
    pub(crate) fn coordinate_descent_config(&self) -> SKCoordinateDescentConfig<F> {
        coordinate_descent_config(
            self.tolerance,
            self.maximum_number_of_iterations,
            self.positive,
            self.selection,
            self.screening,
        )
    }

    /// Fit on a sequential supervised source, retaining each batch's design and
    /// solving the whole accumulated design once at the end.
    ///
    /// The result equals the dense `fit` on the concatenated data within
    /// floating point tolerance, regardless of batch order or size.
    pub fn fit_streaming<S>(&self, source: &mut S) -> Result<SKLassoModel<F>, SKError>
    where
        S: sciencekit_common::SKSupervisedLazySource<F> + Send,
        S::Error: From<SKError> + Into<SKError>,
    {
        let core = crate::coordinate_descent::sk_fit_streaming(
            source,
            &self.alphas,
            F::one(),
            self.include_intercept,
            self.backend,
            self.execution_intent(),
            self.coordinate_descent_config(),
        )?;
        Ok(SKLassoModel::from_core(core))
    }
}

impl<F: SKFloat> SKSupervisedFit<F> for SKLasso<F> {
    type Model = SKLassoModel<F>;
    type Error = SKError;

    fn fit<'a, X, T>(&self, features: X, targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<sciencekit_common::SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        let data = features.try_into()?;
        let target = targets.try_into()?.as_continuous()?.into_owned();
        let core = sk_fit_coordinate_descent(
            data,
            target,
            &self.alphas,
            F::one(),
            self.include_intercept,
            self.backend,
            self.execution_intent(),
            self.coordinate_descent_config(),
        )?;
        Ok(SKLassoModel::from_core(core))
    }
}
