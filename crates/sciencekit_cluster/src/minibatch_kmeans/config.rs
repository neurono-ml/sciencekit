//! MiniBatchKMeans configuration vocabulary (spec `mini-batch-k-means`).

use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};

use crate::kmeans::SKKMeansInitialization;

/// The validated configuration carried by a built [`super::SKMiniBatchKMeans`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct SKMiniBatchKMeansConfiguration<F: SKFloat> {
    /// Requested cluster count.
    pub(crate) number_of_clusters: usize,
    /// Centroid initialization strategy.
    pub(crate) initialization: SKKMeansInitialization,
    /// Number of independent seeded initializations.
    pub(crate) number_of_initializations: usize,
    /// Mini-batch row count for in-memory chunking.
    pub(crate) batch_size: usize,
    /// Maximum sample size drawn for centroid initialization.
    pub(crate) initialization_batch_size: usize,
    /// Maximum number of passes over in-memory training data.
    pub(crate) maximum_number_of_iterations: usize,
    /// Maximum number of streaming passes over a reopenable source provider.
    pub(crate) maximum_number_of_epochs: usize,
    /// Convergence tolerance (`0` disables early stopping).
    pub(crate) tolerance: F,
    /// Low-count center reassignment threshold fraction in `[0, 1]`.
    pub(crate) reassignment_ratio: F,
    /// Deterministic global seed.
    pub(crate) random_seed: u64,
    /// Math backend recorded in operation spans.
    pub(crate) backend: SKBackendKind,
    /// Execution intent.
    pub(crate) execution_mode: SKExecutionMode,
}
