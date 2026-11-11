//! The generic bagging regressor configuration estimator.

use std::error::Error;

use sciencekit_common::{
    SKBackendKind, SKDataView, SKError, SKExecutionContext, SKExecutionMode, SKExecutionPlan,
    SKFloat, SKRegressorPredictor, SKSupervisedFit, SKTargetView,
};
use sciencekit_tree::SKMaximumFeatures;

use crate::bootstrap::SKMaximumSamples;

use super::builder::SKBaggingRegressorBuilder;
use super::config::SKBaggingConfiguration;
use super::error::SKBaggingError;
use super::model::SKBaggingRegressorModel;

/// Generic bootstrap-aggregating regressor over base estimator `B`,
/// constructed only through its builder.
///
/// The base estimator's fitted model must implement [`SKRegressorPredictor`];
/// passing a classifier configuration fails to compile when fitting:
///
/// ```compile_fail
/// use ndarray::{Array1, Array2};
/// use sciencekit_common::{SKBuilder, SKSupervisedFit};
/// use sciencekit_ensemble::SKBaggingRegressor;
/// use sciencekit_tree::SKDecisionTreeClassifier;
///
/// let classifier = SKDecisionTreeClassifier::<f64>::build().build().unwrap();
/// let bagging = SKBaggingRegressor::build(classifier).build().unwrap();
/// let features = Array2::<f64>::zeros((4, 2));
/// let targets = Array1::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0]);
/// let _ = bagging.fit(features.view(), targets.view());
/// ```
#[derive(Debug, Clone)]
pub struct SKBaggingRegressor<B> {
    pub(super) base_estimator: B,
    pub(super) configuration: SKBaggingConfiguration,
}

impl<B> SKBaggingRegressor<B> {
    /// Start building a bagging regressor around `base_estimator`.
    pub fn build(base_estimator: B) -> SKBaggingRegressorBuilder<B> {
        SKBaggingRegressorBuilder::new(base_estimator)
    }

    /// Borrow the configured base estimator template.
    pub fn base_estimator(&self) -> &B {
        &self.base_estimator
    }

    /// The configured number of base estimators (default `10`).
    pub fn number_of_estimators(&self) -> usize {
        self.configuration.number_of_estimators
    }

    /// The per-estimator row-subsampling policy (default `All`).
    pub fn maximum_samples(&self) -> &SKMaximumSamples<f64> {
        &self.configuration.maximum_samples
    }

    /// The per-estimator feature-subsampling policy (default `All`).
    pub fn maximum_features(&self) -> &SKMaximumFeatures<f64> {
        &self.configuration.maximum_features
    }

    /// Whether rows are sampled with replacement (default `true`).
    pub fn bootstrap(&self) -> bool {
        self.configuration.bootstrap
    }

    /// Whether features are sampled with replacement (default `false`).
    pub fn bootstrap_features(&self) -> bool {
        self.configuration.bootstrap_features
    }

    /// Whether out-of-bag R² estimation is enabled (default `false`).
    pub fn oob_score(&self) -> bool {
        self.configuration.oob_score
    }

    /// The global seed from which per-estimator seeds are derived (default `0`).
    pub fn random_seed(&self) -> u64 {
        self.configuration.random_seed
    }

    /// The backend kind recorded in operation spans and transferred to models.
    pub fn backend(&self) -> SKBackendKind {
        self.configuration.backend
    }

    /// The configured execution intent.
    pub fn execution_intent(&self) -> SKExecutionMode {
        self.configuration.execution_intent()
    }

    /// Resolve the execution intent against an execution context.
    pub fn resolve_plan(&self, context: &SKExecutionContext) -> Result<SKExecutionPlan, SKError> {
        self.configuration.state.resolve_plan(context)
    }
}

impl<F, B> SKSupervisedFit<F> for SKBaggingRegressor<B>
where
    F: SKFloat,
    B: SKSupervisedFit<F> + Clone + Send + Sync,
    B::Model: SKRegressorPredictor<F> + Send + Sync,
    B::Error: Error + Send + 'static,
    <B::Model as SKRegressorPredictor<F>>::Error: Error + Send + 'static,
{
    type Model = SKBaggingRegressorModel<F, B::Model>;
    type Error = SKBaggingError<B::Error>;

    fn fit<'a, X, T>(&self, features: X, targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        super::fit::sk_fit_bagging_regressor(self, features.try_into()?, targets.try_into()?)
    }
}
