//! Fitted Incremental Principal Component Analysis model
//! (`SKIncrementalPCAModel`, spec `incremental-pca`).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};

use super::accumulator::SKCovarianceAccumulator;
use super::config::SKIncrementalPCAConfiguration;
use crate::pca::SKPCADecomposition;

/// Fitted Incremental Principal Component Analysis state: running covariance
/// accumulator, finalized principal components, singular values, explained
/// variance and whitening scales.
#[derive(Debug, Clone)]
pub struct SKIncrementalPCAModel<F: SKFloat> {
    pub(super) accumulator: SKCovarianceAccumulator<F>,
    pub(super) components: Array2<F>,
    pub(super) explained_variance: Array1<F>,
    pub(super) explained_variance_ratio: Array1<F>,
    pub(super) singular_values: Array1<F>,
    pub(super) whiten_scale: Array1<F>,
    pub(super) epochs_consumed: usize,
    pub(super) configuration: SKIncrementalPCAConfiguration,
}

impl<F: SKFloat> SKIncrementalPCAModel<F> {
    pub(super) fn from_accumulator_and_decomposition(
        accumulator: SKCovarianceAccumulator<F>,
        decomposition: SKPCADecomposition<F>,
        epochs_consumed: usize,
        configuration: SKIncrementalPCAConfiguration,
    ) -> Self {
        let whiten_scale = sk_whiten_scales(&decomposition.explained_variance);
        Self {
            accumulator,
            components: decomposition.components,
            explained_variance: decomposition.explained_variance,
            explained_variance_ratio: decomposition.explained_variance_ratio,
            singular_values: decomposition.singular_values,
            whiten_scale,
            epochs_consumed,
            configuration,
        }
    }

    pub(super) fn refresh_decomposition(&mut self, decomposition: SKPCADecomposition<F>) {
        self.whiten_scale = sk_whiten_scales(&decomposition.explained_variance);
        self.components = decomposition.components;
        self.explained_variance = decomposition.explained_variance;
        self.explained_variance_ratio = decomposition.explained_variance_ratio;
        self.singular_values = decomposition.singular_values;
    }

    /// Principal axes in feature space, with shape `k × d`.
    pub fn components(&self) -> ArrayView2<'_, F> {
        self.components.view()
    }

    /// Unbiased variance explained by each retained component, non-increasing.
    pub fn explained_variance(&self) -> ArrayView1<'_, F> {
        self.explained_variance.view()
    }

    /// Fraction of total centered variance explained by each component.
    pub fn explained_variance_ratio(&self) -> ArrayView1<'_, F> {
        self.explained_variance_ratio.view()
    }

    /// Singular values corresponding to each retained component, non-increasing.
    pub fn singular_values(&self) -> ArrayView1<'_, F> {
        self.singular_values.view()
    }

    /// Running empirical per-feature mean across all processed batches.
    pub fn mean(&self) -> ArrayView1<'_, F> {
        self.accumulator.mean()
    }

    /// Running centered scatter matrix `Σ (x − mean)(x − mean)ᵀ` of shape `d × d`.
    pub fn scatter_matrix(&self) -> ArrayView2<'_, F> {
        self.accumulator.scatter()
    }

    /// Total number of rows accumulated across `fit`, `partial_fit` or
    /// `fit_streaming`.
    pub fn samples_seen(&self) -> usize {
        self.accumulator.samples_seen()
    }

    /// Number of streaming passes consumed (`1` for in-memory `fit`).
    pub fn epochs_consumed(&self) -> usize {
        self.epochs_consumed
    }

    /// Number of retained principal components `k`.
    pub fn number_of_components(&self) -> usize {
        self.components.nrows()
    }

    /// Number of input features `d` seen during fitting.
    pub fn number_of_features(&self) -> usize {
        self.components.ncols()
    }

    /// Configured row batch size transferred from the estimator.
    pub fn batch_size(&self) -> usize {
        self.configuration.batch_size
    }

    /// Whether `transform` scales projections to unit variance.
    pub fn whiten(&self) -> bool {
        self.configuration.whiten
    }

    /// Backend kind transferred from the estimator and recorded in model spans.
    pub fn backend(&self) -> SKBackendKind {
        self.configuration.backend
    }

    /// Execution intent transferred from the estimator.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.configuration.execution_mode()
    }
}

fn sk_whiten_scales<F: SKFloat>(explained_variance: &Array1<F>) -> Array1<F> {
    let floor = F::epsilon();
    explained_variance.mapv(|variance| variance.max(floor).sqrt())
}
