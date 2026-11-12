//! Configured Truncated SVD estimator (`SKTruncatedSVD`, spec `truncated-svd`).

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionContext, SKExecutionMode, SKExecutionPlan, SKFloat,
};

use super::builder::SKTruncatedSVDBuilder;
use super::config::{SKTruncatedSVDAlgorithm, SKTruncatedSVDConfiguration};

/// Truncated Singular Value Decomposition, constructible only through
/// [`SKTruncatedSVDBuilder`].
///
/// `SKTruncatedSVD` computes an uncentered low-rank factorization of dense or
/// sparse CSR matrices using the seeded randomized range finder, never
/// densifying sparse inputs. Fitting produces a distinct
/// [`super::SKTruncatedSVDModel`] holding the right singular vectors, singular
/// values, explained variance and explained-variance ratio.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SKTruncatedSVD<F: SKFloat> {
    pub(super) configuration: SKTruncatedSVDConfiguration<F>,
}

impl<F: SKFloat> SKTruncatedSVD<F> {
    /// Opens a [`SKTruncatedSVDBuilder`] initialized with the documented
    /// defaults.
    pub fn build() -> SKTruncatedSVDBuilder<F> {
        SKTruncatedSVDBuilder::new()
    }

    /// Configured number of retained singular components `k` (default `2`).
    pub fn number_of_components(&self) -> usize {
        self.configuration.number_of_components
    }

    /// Requested low-rank SVD solver (`Randomized` by default).
    pub fn algorithm(&self) -> SKTruncatedSVDAlgorithm {
        self.configuration.algorithm
    }

    /// Number of power iterations for the randomized range finder (default `5`).
    pub fn number_of_iterations(&self) -> usize {
        self.configuration.number_of_iterations
    }

    /// Configured solver convergence tolerance (default `0.0`; ignored by
    /// `Randomized`).
    pub fn tolerance(&self) -> F {
        self.configuration.tolerance
    }

    /// Deterministic seed for the randomized projection (default `0`).
    pub fn random_seed(&self) -> u64 {
        self.configuration.random_seed
    }

    /// Backend kind recorded in operation spans and transferred to fitted
    /// models (`Faer` by default).
    pub fn backend(&self) -> SKBackendKind {
        self.configuration.backend
    }

    /// Configured execution intent (`Automatic` by default).
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.configuration.execution_mode()
    }

    /// Resolves the estimator's execution intent against `context`.
    pub fn resolve_plan(&self, context: &SKExecutionContext) -> Result<SKExecutionPlan, SKError> {
        self.configuration.builder_state.resolve_plan(context)
    }

    pub(super) fn from_configuration(configuration: SKTruncatedSVDConfiguration<F>) -> Self {
        Self { configuration }
    }
}
