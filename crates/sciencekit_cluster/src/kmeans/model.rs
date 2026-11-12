//! The fitted KMeans model (spec `k-means`, decision 8).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sciencekit_common::{SKBackendKind, SKClusterLabel, SKExecutionMode, SKFloat};

use super::SKKMeansAlgorithm;
use super::result::SKKMeansFitResult;

/// Fitted KMeans state: dense centers, typed training assignments, inertia,
/// iteration count, convergence flag and the metadata of the producing run.
#[derive(Debug, Clone)]
pub struct SKKMeansModel<F: SKFloat> {
    pub(super) centers: Array2<F>,
    pub(super) labels: Array1<SKClusterLabel>,
    pub(super) inertia: F,
    pub(super) iterations: usize,
    pub(super) converged: bool,
    pub(super) algorithm: SKKMeansAlgorithm,
    pub(super) backend: SKBackendKind,
    pub(super) execution_mode: SKExecutionMode,
}

impl<F: SKFloat> SKKMeansModel<F> {
    /// Builds the public model from a finished initialization.
    pub(super) fn from_fit(
        result: SKKMeansFitResult<F>,
        labels: Array1<SKClusterLabel>,
        algorithm: SKKMeansAlgorithm,
        backend: SKBackendKind,
        execution_mode: SKExecutionMode,
    ) -> Self {
        Self {
            centers: result.centers,
            labels,
            inertia: result.inertia,
            iterations: result.iterations,
            converged: result.converged,
            algorithm,
            backend,
            execution_mode,
        }
    }

    /// Cluster centers, one row per cluster.
    pub fn cluster_centers(&self) -> ArrayView2<'_, F> {
        self.centers.view()
    }

    /// Typed training assignment, one `Cluster` variant per training row.
    pub fn labels(&self) -> ArrayView1<'_, SKClusterLabel> {
        self.labels.view()
    }

    /// Sum of squared distances from every training row to its assigned
    /// center.
    pub fn inertia(&self) -> F {
        self.inertia
    }

    /// Iterations executed by the winning initialization.
    pub fn iterations_taken(&self) -> usize {
        self.iterations
    }

    /// Whether the center movement met the tolerance within the budget.
    pub fn converged(&self) -> bool {
        self.converged
    }

    /// Number of clusters in the model.
    pub fn number_of_clusters(&self) -> usize {
        self.centers.nrows()
    }

    /// Number of features the model was fitted on.
    pub fn number_of_features(&self) -> usize {
        self.centers.ncols()
    }

    /// Resolved assignment algorithm (never `Auto`).
    pub fn algorithm(&self) -> SKKMeansAlgorithm {
        self.algorithm
    }

    /// Backend kind recorded by the producing estimator and by every model
    /// operation.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// Execution intent carried from the estimator.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.execution_mode
    }
}
