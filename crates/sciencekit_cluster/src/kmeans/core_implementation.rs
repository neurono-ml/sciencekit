//! The configured KMeans estimator (spec `k-means`).

use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};

use super::SKKMeansAlgorithm;
use super::builder::SKKMeansBuilder;
use super::config::{SKKMeansConfiguration, SKKMeansInitialization};

/// Lloyd/Elkan KMeans, constructed only through [`SKKMeansBuilder`].
///
/// The estimator is immutable configuration; fitting it returns the distinct
/// fitted `SKKMeansModel`, which owns the learned centers and assignments.
#[derive(Debug, Clone, Copy)]
pub struct SKKMeans<F: SKFloat> {
    pub(super) configuration: SKKMeansConfiguration<F>,
}

impl<F: SKFloat> SKKMeans<F> {
    /// Starts the mandatory builder with the documented defaults.
    pub fn build() -> SKKMeansBuilder<F> {
        SKKMeansBuilder::new()
    }

    /// Number of clusters.
    pub fn number_of_clusters(&self) -> usize {
        self.configuration.number_of_clusters
    }

    /// Centroid initialization strategy.
    pub fn initialization(&self) -> SKKMeansInitialization {
        self.configuration.initialization
    }

    /// Number of independent initializations, after default resolution.
    pub fn number_of_initializations(&self) -> usize {
        self.configuration.number_of_initializations
    }

    /// Iteration budget per initialization.
    pub fn maximum_number_of_iterations(&self) -> usize {
        self.configuration.maximum_number_of_iterations
    }

    /// Convergence tolerance.
    pub fn tolerance(&self) -> F {
        self.configuration.tolerance
    }

    /// Requested assignment algorithm (may be `Auto`).
    pub fn algorithm(&self) -> SKKMeansAlgorithm {
        self.configuration.algorithm
    }

    /// Global seed of the deterministic randomness.
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

    /// Builds the estimator from an already validated configuration.
    pub(super) fn from_configuration(configuration: SKKMeansConfiguration<F>) -> Self {
        Self { configuration }
    }
}
