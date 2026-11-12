//! The histogram-ensemble gradient-boosting regressor configuration.

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionContext, SKExecutionMode, SKExecutionPlan, SKFloat,
};
use sciencekit_tree::SKMaximumFeatures;

use super::regressor_builder::SKHistGradientBoostingRegressorBuilder;
use super::config::SKHistogramGradientBoostingConfiguration;
use super::loss_kind::SKHistGradientBoostingRegressionLoss;

/// Histogram gradient-boosting regressor, constructed only through
/// [`SKHistGradientBoostingRegressor::build`].
#[derive(Debug, Clone)]
pub struct SKHistGradientBoostingRegressor<F> {
    pub(super) configuration: SKHistogramGradientBoostingConfiguration<F>,
    pub(super) loss: SKHistGradientBoostingRegressionLoss,
}

impl<F: SKFloat> SKHistGradientBoostingRegressor<F> {
    /// Start building a histogram gradient-boosting regressor.
    pub fn build() -> SKHistGradientBoostingRegressorBuilder<F> {
        SKHistGradientBoostingRegressorBuilder::new()
    }

    /// Configured maximum number of boosting stages (default `100`).
    pub fn number_of_estimators(&self) -> usize {
        self.configuration.number_of_estimators
    }

    /// Configured regression loss (`SquaredError` by default).
    pub fn loss(&self) -> SKHistGradientBoostingRegressionLoss {
        self.loss
    }

    /// Shrinkage factor applied to every Newton leaf (default `0.1`).
    pub fn learning_rate(&self) -> F {
        self.configuration.learning_rate
    }

    /// Maximum number of quantile bins per feature (default `255`).
    pub fn maximum_bins(&self) -> usize {
        self.configuration.maximum_bins
    }

    /// Maximum number of leaves per stage tree (default `31`).
    pub fn maximum_leaf_nodes(&self) -> usize {
        self.configuration.maximum_leaf_nodes
    }

    /// Minimum samples required in every leaf (default `20`).
    pub fn minimum_samples_leaf(&self) -> usize {
        self.configuration.minimum_samples_leaf
    }

    /// L2 Hessian damping term `λ` (default `0.0`).
    pub fn l2_regularization(&self) -> F {
        self.configuration.l2_regularization
    }

    /// Per-split feature-subsampling policy.
    pub fn maximum_features(&self) -> &SKMaximumFeatures<F> {
        &self.configuration.maximum_features
    }

    /// Column indices treated as categorical.
    pub fn categorical_features(&self) -> &[usize] {
        &self.configuration.categorical_features
    }

    /// Row-subsampling fraction drawn without replacement per stage.
    pub fn subsample(&self) -> F {
        self.configuration.subsample
    }

    /// Global deterministic seed.
    pub fn random_seed(&self) -> u64 {
        self.configuration.random_seed
    }

    /// Held-out fraction used by early stopping (default `0.1`).
    pub fn validation_fraction(&self) -> F {
        self.configuration.validation_fraction
    }

    /// Optional non-improving stage patience (default `Some(10)`).
    pub fn n_iter_no_change(&self) -> Option<usize> {
        self.configuration.n_iter_no_change
    }

    /// Minimum validation-loss improvement required to reset patience.
    pub fn tolerance(&self) -> F {
        self.configuration.tolerance
    }

    /// Configured backend kind recorded in spans and transferred to models.
    pub fn backend(&self) -> SKBackendKind {
        self.configuration.backend
    }

    /// Configured execution intent.
    pub fn execution_intent(&self) -> SKExecutionMode {
        self.configuration.state.execution_intent()
    }

    /// Resolve the execution intent against an execution context.
    pub fn resolve_plan(&self, context: &SKExecutionContext) -> Result<SKExecutionPlan, SKError> {
        self.configuration.state.resolve_plan(context)
    }
}
