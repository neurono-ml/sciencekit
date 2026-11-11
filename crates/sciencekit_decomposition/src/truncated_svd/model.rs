//! Fitted Truncated SVD model (`SKTruncatedSVDModel`, spec `truncated-svd`).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sciencekit_common::{SKBackendKind, SKBuilderState, SKExecutionMode, SKFloat};

use super::config::SKTruncatedSVDAlgorithm;

/// Fitted uncentered low-rank factorization state: right singular vectors
/// (`components`), singular values, explained variance and explained-variance
/// ratio. Stores no centering vector.
#[derive(Debug, Clone)]
pub struct SKTruncatedSVDModel<F: SKFloat> {
    pub(super) components: Array2<F>,
    pub(super) singular_values: Array1<F>,
    pub(super) explained_variance: Array1<F>,
    pub(super) explained_variance_ratio: Array1<F>,
    pub(super) algorithm: SKTruncatedSVDAlgorithm,
    pub(super) backend: SKBackendKind,
    pub(super) builder_state: SKBuilderState,
}

impl<F: SKFloat> SKTruncatedSVDModel<F> {
    #[allow(dead_code, clippy::too_many_arguments)]
    pub(super) fn from_parts(
        components: Array2<F>,
        singular_values: Array1<F>,
        explained_variance: Array1<F>,
        explained_variance_ratio: Array1<F>,
        algorithm: SKTruncatedSVDAlgorithm,
        backend: SKBackendKind,
        builder_state: SKBuilderState,
    ) -> Self {
        Self {
            components,
            singular_values,
            explained_variance,
            explained_variance_ratio,
            algorithm,
            backend,
            builder_state,
        }
    }

    /// Right singular vectors (`Vᵀ`) in feature space, with shape `k × d`.
    pub fn components(&self) -> ArrayView2<'_, F> {
        self.components.view()
    }

    /// Retained singular values, non-increasing.
    pub fn singular_values(&self) -> ArrayView1<'_, F> {
        self.singular_values.view()
    }

    /// Population variance of each projected component over the training rows.
    pub fn explained_variance(&self) -> ArrayView1<'_, F> {
        self.explained_variance.view()
    }

    /// Fraction of total input column variance captured by each component.
    pub fn explained_variance_ratio(&self) -> ArrayView1<'_, F> {
        self.explained_variance_ratio.view()
    }

    /// Number of retained components `k`.
    pub fn number_of_components(&self) -> usize {
        self.components.nrows()
    }

    /// Number of input features `d` seen during `fit`.
    pub fn number_of_features(&self) -> usize {
        self.components.ncols()
    }

    /// Solver algorithm executed during `fit`.
    pub fn algorithm(&self) -> SKTruncatedSVDAlgorithm {
        self.algorithm
    }

    /// Backend kind transferred from the estimator and recorded in model spans.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// Execution intent transferred from the estimator.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.builder_state.execution_intent()
    }
}
