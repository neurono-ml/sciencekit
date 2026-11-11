//! Configured `SKIsolationForest` estimator type.

use sciencekit_common::{SKBackendKind, SKBuilderState, SKExecutionMode, SKFloat};

use super::builder::SKIsolationForestBuilder;
use super::config::{SKContamination, SKIsolationMaximumSamples};

/// Configured Isolation Forest outlier detector (spec `isolation-forest`).
///
/// Constructed exclusively via [`SKIsolationForest::build`].
#[derive(Debug, Clone)]
pub struct SKIsolationForest<F: SKFloat> {
    pub(super) state: SKBuilderState,
    pub(super) number_of_trees: usize,
    pub(super) maximum_samples: SKIsolationMaximumSamples<F>,
    pub(super) contamination: SKContamination<F>,
    pub(super) maximum_features: F,
    pub(super) bootstrap: bool,
    pub(super) random_seed: u64,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKIsolationForest<F> {
    /// Entry point to the mandatory builder.
    pub fn build() -> SKIsolationForestBuilder<F> {
        SKIsolationForestBuilder::new()
    }

    /// Configured number of isolation trees.
    pub fn number_of_trees(&self) -> usize {
        self.number_of_trees
    }

    /// Configured per-tree subsample policy.
    pub fn maximum_samples(&self) -> SKIsolationMaximumSamples<F> {
        self.maximum_samples
    }

    /// Configured contamination policy.
    pub fn contamination(&self) -> SKContamination<F> {
        self.contamination
    }

    /// Configured fraction of features considered per tree in `(0, 1]`.
    pub fn maximum_features(&self) -> F {
        self.maximum_features
    }

    /// Whether subsampling draws with replacement.
    pub fn bootstrap(&self) -> bool {
        self.bootstrap
    }

    /// Deterministic global random seed.
    pub fn random_seed(&self) -> u64 {
        self.random_seed
    }

    /// Configured backend kind recorded on observability spans.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// Configured execution intent.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.state.execution_intent()
    }
}
