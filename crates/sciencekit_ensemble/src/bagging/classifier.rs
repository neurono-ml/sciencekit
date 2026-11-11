//! The generic bagging classifier configuration estimator.

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionContext, SKExecutionMode, SKExecutionPlan,
};
use sciencekit_tree::SKMaximumFeatures;

use crate::bootstrap::SKMaximumSamples;

use super::builder::SKBaggingClassifierBuilder;
use super::config::SKBaggingConfiguration;

/// Generic bootstrap-aggregating classifier over base estimator `B`,
/// constructed only through its builder.
#[derive(Debug, Clone)]
pub struct SKBaggingClassifier<B> {
    pub(super) base_estimator: B,
    pub(super) configuration: SKBaggingConfiguration,
}

impl<B> SKBaggingClassifier<B> {
    /// Start building a bagging classifier around `base_estimator`.
    pub fn build(base_estimator: B) -> SKBaggingClassifierBuilder<B> {
        SKBaggingClassifierBuilder::new(base_estimator)
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

    /// Whether out-of-bag accuracy estimation is enabled (default `false`).
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
