//! The result of one Lloyd or Elkan run (spec `k-means`).

use ndarray::{Array1, Array2};
use sciencekit_common::SKFloat;

/// Learned state of one k-means run, before it becomes a public model.
pub(crate) struct SKKMeansFitResult<F: SKFloat> {
    /// One row per cluster.
    pub(crate) centers: Array2<F>,
    /// Zero-based training assignment per row.
    pub(crate) labels: Array1<usize>,
    /// Sum of squared distances to the assigned centers.
    pub(crate) inertia: F,
    /// Iterations actually executed.
    pub(crate) iterations: usize,
    /// Whether the center movement met the tolerance.
    pub(crate) converged: bool,
}
