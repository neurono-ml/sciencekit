//! The `SKElasticNet` estimator and its supervised fit (spec `elastic-net`).

use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKError, SKExecutionContext, SKExecutionMode, SKExecutionPlan,
    SKFloat, SKSupervisedFit, SKTargetView,
};

use crate::coordinate_descent::SKCoordinateDescentConfig;
use crate::coordinate_descent::SKSelection;
use crate::coordinate_descent::sk_fit_coordinate_descent;
use crate::elastic_net::builder::SKElasticNetBuilder;
use crate::elastic_net::model::SKElasticNetModel;

/// Elastic net regression, builder-configured.
///
/// Immutable configuration carrying scalar values (`alpha`, `l1_ratio`,
/// `tolerance`), so the estimator is generic over the model scalar `F`. The
/// learned state lives only on the distinct [`SKElasticNetModel`].
#[derive(Debug, Clone)]
pub struct SKElasticNet<F> {
    pub(super) plan: SKBuilderState,
    pub(super) alphas: Vec<F>,
    pub(super) l1_ratio: F,
    pub(super) include_intercept: bool,
    pub(super) config: SKCoordinateDescentConfig<F>,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKElasticNet<F> {
    /// Start building an estimator (the only construction surface).
    pub fn build() -> SKElasticNetBuilder<F> {
        SKElasticNetBuilder::new()
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
        self.config.positive
    }

    /// The coordinate sweep order.
    pub fn selection(&self) -> SKSelection {
        self.config.selection
    }

    /// Whether gap-safe screening is enabled.
    pub fn screening(&self) -> bool {
        self.config.screening
    }

    /// The configured L1/L2 mixing ratio.
    pub fn l1_ratio(&self) -> F {
        self.l1_ratio
    }

    /// The configured penalty values, in configuration order.
    pub fn alphas(&self) -> &[F] {
        &self.alphas
    }

    /// The convergence tolerance.
    pub fn tolerance(&self) -> F {
        self.config.tolerance
    }

    /// The iteration budget.
    pub fn maximum_number_of_iterations(&self) -> usize {
        self.config.maximum_number_of_iterations
    }

    /// Fit on a sequential supervised source, retaining each batch's design and
    /// solving the whole accumulated design once at the end.
    ///
    /// The result equals the dense `fit` on the concatenated data within
    /// floating point tolerance, regardless of batch order or size.
    pub fn fit_streaming<S>(&self, source: &mut S) -> Result<SKElasticNetModel<F>, SKError>
    where
        S: sciencekit_common::SKSupervisedLazySource<F> + Send,
        S::Error: From<SKError> + Into<SKError>,
    {
        let core = crate::coordinate_descent::sk_fit_streaming(
            source,
            &self.alphas,
            self.l1_ratio,
            self.include_intercept,
            self.backend,
            self.execution_intent(),
            self.config.clone(),
        )?;
        Ok(SKElasticNetModel::from_core(core))
    }
}

impl<F: SKFloat> SKSupervisedFit<F> for SKElasticNet<F> {
    type Model = SKElasticNetModel<F>;
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
            self.l1_ratio,
            self.include_intercept,
            self.backend,
            self.execution_intent(),
            self.config.clone(),
        )?;
        Ok(SKElasticNetModel::from_core(core))
    }
}
