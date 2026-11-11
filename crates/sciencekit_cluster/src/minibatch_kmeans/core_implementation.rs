//! The configured MiniBatchKMeans estimator (spec `mini-batch-k-means`).

use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};

use super::builder::SKMiniBatchKMeansBuilder;
use super::config::SKMiniBatchKMeansConfiguration;
use crate::kmeans::SKKMeansInitialization;

/// Mini-batch KMeans with per-center learning rates and out-of-core streaming,
/// constructed only through [`SKMiniBatchKMeansBuilder`].
#[derive(Debug, Clone, Copy)]
pub struct SKMiniBatchKMeans<F: SKFloat> {
    pub(super) configuration: SKMiniBatchKMeansConfiguration<F>,
}

impl<F: SKFloat> SKMiniBatchKMeans<F> {
    /// Starts the mandatory builder with the documented defaults.
    pub fn build() -> SKMiniBatchKMeansBuilder<F> {
        SKMiniBatchKMeansBuilder::new()
    }

    /// Requested cluster count.
    pub fn number_of_clusters(&self) -> usize {
        self.configuration.number_of_clusters
    }

    /// Centroid initialization strategy.
    pub fn initialization(&self) -> SKKMeansInitialization {
        self.configuration.initialization
    }

    /// Number of independent seeded initializations.
    pub fn number_of_initializations(&self) -> usize {
        self.configuration.number_of_initializations
    }

    /// Mini-batch row count for in-memory chunking.
    pub fn batch_size(&self) -> usize {
        self.configuration.batch_size
    }

    /// Maximum sample size drawn for centroid initialization.
    pub fn initialization_batch_size(&self) -> usize {
        self.configuration.initialization_batch_size
    }

    /// Iteration (pass) budget for in-memory `fit`.
    pub fn maximum_number_of_iterations(&self) -> usize {
        self.configuration.maximum_number_of_iterations
    }

    /// Epoch budget for `fit_streaming` over reopenable providers.
    pub fn maximum_number_of_epochs(&self) -> usize {
        self.configuration.maximum_number_of_epochs
    }

    /// Convergence tolerance (`0` disables early stopping).
    pub fn tolerance(&self) -> F {
        self.configuration.tolerance
    }

    /// Low-count center reassignment threshold fraction in `[0, 1]`.
    pub fn reassignment_ratio(&self) -> F {
        self.configuration.reassignment_ratio
    }

    /// Global deterministic seed.
    pub fn random_seed(&self) -> u64 {
        self.configuration.random_seed
    }

    /// Backend kind recorded in every operation span.
    pub fn backend(&self) -> SKBackendKind {
        self.configuration.backend
    }

    /// Configured execution intent.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.configuration.execution_mode
    }

    /// Private constructor from a validated configuration.
    pub(super) fn from_configuration(configuration: SKMiniBatchKMeansConfiguration<F>) -> Self {
        Self { configuration }
    }
}
