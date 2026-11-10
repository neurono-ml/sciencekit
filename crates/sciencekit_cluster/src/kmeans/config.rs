//! KMeans configuration vocabulary (spec `k-means`).

use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};

/// Centroid initialization strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKKMeansInitialization {
    /// D² sampling: the first center is uniform, each next center is drawn with
    /// probability proportional to its squared distance to the nearest chosen
    /// center.
    KMeansPlusPlus,
    /// Uniformly drawn distinct training rows.
    Random,
}

impl Default for SKKMeansInitialization {
    fn default() -> Self {
        Self::KMeansPlusPlus
    }
}

/// Assignment algorithm requested by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKKMeansAlgorithm {
    /// Elkan for dense input, Lloyd for sparse input.
    Auto,
    /// Plain Lloyd iteration.
    Lloyd,
    /// Triangle-inequality Elkan acceleration (dense input only).
    Elkan,
}

impl Default for SKKMeansAlgorithm {
    fn default() -> Self {
        Self::Auto
    }
}

/// The validated configuration carried by a built estimator.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SKKMeansConfiguration<F: SKFloat> {
    /// Requested cluster count.
    pub(crate) number_of_clusters: usize,
    /// Centroid initialization strategy.
    pub(crate) initialization: SKKMeansInitialization,
    /// Number of independent initializations.
    pub(crate) number_of_initializations: usize,
    /// Iteration budget per initialization.
    pub(crate) maximum_number_of_iterations: usize,
    /// Convergence tolerance.
    pub(crate) tolerance: F,
    /// Requested assignment algorithm.
    pub(crate) algorithm: SKKMeansAlgorithm,
    /// Global seeded randomness.
    pub(crate) random_seed: u64,
    /// Math backend recorded in operation spans.
    pub(crate) backend: SKBackendKind,
    /// Execution intent.
    pub(crate) execution_mode: SKExecutionMode,
}
