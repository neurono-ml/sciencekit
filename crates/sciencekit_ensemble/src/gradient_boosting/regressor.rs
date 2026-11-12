//! The gradient-boosting regressor configuration estimator.

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionContext, SKExecutionMode, SKExecutionPlan, SKFloat,
};
use sciencekit_tree::SKMaximumFeatures;

use super::config::SKGradientBoostingConfiguration;
use super::loss_kind::SKGradientBoostingRegressionLoss;
use super::regressor_builder::SKGradientBoostingRegressorBuilder;

/// Stage-wise additive gradient-boosting regressor, constructed only through
/// [`SKGradientBoostingRegressor::build`].
#[derive(Debug, Clone)]
pub struct SKGradientBoostingRegressor<F> {
    pub(super) configuration: SKGradientBoostingConfiguration<F>,
    pub(super) loss: SKGradientBoostingRegressionLoss,
    pub(super) huber_delta: F,
}

impl<F: SKFloat> SKGradientBoostingRegressor<F> {
    /// Start building a gradient-boosting regressor (sole construction surface).
    pub fn build() -> SKGradientBoostingRegressorBuilder<F> {
        SKGradientBoostingRegressorBuilder::new()
    }

    /// Configured maximum number of boosting stages.
    pub fn number_of_estimators(&self) -> usize {
        self.configuration.number_of_estimators
    }

    /// Shrinkage factor applied to every Newton leaf step.
    pub fn learning_rate(&self) -> F {
        self.configuration.learning_rate
    }

    /// Configured regression loss (`SquaredError`, `AbsoluteError` or `Huber`).
    pub fn loss(&self) -> SKGradientBoostingRegressionLoss {
        self.loss
    }

    /// Clipping threshold `delta` used when `loss` is `Huber`.
    pub fn huber_delta(&self) -> F {
        self.huber_delta
    }

    /// Maximum depth of every stage tree.
    pub fn maximum_depth(&self) -> usize {
        self.configuration.maximum_depth
    }

    /// Minimum sample count required in every stage leaf.
    pub fn minimum_samples_leaf(&self) -> usize {
        self.configuration.minimum_samples_leaf
    }

    /// Per-split feature-subsampling policy.
    pub fn maximum_features(&self) -> &SKMaximumFeatures<F> {
        &self.configuration.maximum_features
    }

    /// Row-subsampling fraction drawn without replacement per stage.
    pub fn subsample(&self) -> F {
        self.configuration.subsample
    }

    /// L2 Hessian damping term `λ` in `-Σg / (Σh + λ)`.
    pub fn l2_regularization(&self) -> F {
        self.configuration.l2_regularization
    }

    /// Global deterministic seed from which stage and split seeds are derived.
    pub fn random_seed(&self) -> u64 {
        self.configuration.random_seed
    }

    /// Held-out fraction used when `n_iter_no_change` is enabled.
    pub fn validation_fraction(&self) -> F {
        self.configuration.validation_fraction
    }

    /// Optional non-improving stage patience enabling early stopping.
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
