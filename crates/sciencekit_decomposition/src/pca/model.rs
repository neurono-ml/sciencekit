//! Fitted Principal Component Analysis model (`SKPCAModel`, spec
//! `principal-component-analysis`).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sciencekit_common::{SKBackendKind, SKBuilderState, SKExecutionMode, SKFloat};

use super::config::SKPCASolver;
use super::solvers::SKPCADecomposition;

/// Fitted Principal Component Analysis state: learned feature means, principal
/// components, singular values, explained variance and whitening scales.
#[derive(Debug, Clone)]
pub struct SKPCAModel<F: SKFloat> {
    pub(super) components: Array2<F>,
    pub(super) explained_variance: Array1<F>,
    pub(super) explained_variance_ratio: Array1<F>,
    pub(super) singular_values: Array1<F>,
    pub(super) mean: Array1<F>,
    pub(super) whiten: bool,
    pub(super) resolved_solver: SKPCASolver,
    pub(super) backend: SKBackendKind,
    pub(super) builder_state: SKBuilderState,
}

impl<F: SKFloat> SKPCAModel<F> {
    /// Wraps a finished solver decomposition into the public model, flooring
    /// explained variance at `F::epsilon()` when computing the whitening
    /// divisor so near-zero components never produce `NaN` or `±∞`.
    pub(super) fn from_decomposition(
        decomposition: SKPCADecomposition<F>,
        mean: Array1<F>,
        whiten: bool,
        resolved_solver: SKPCASolver,
        backend: SKBackendKind,
        builder_state: SKBuilderState,
    ) -> Self {
        Self {
            components: decomposition.components,
            explained_variance: decomposition.explained_variance,
            explained_variance_ratio: decomposition.explained_variance_ratio,
            singular_values: decomposition.singular_values,
            mean,
            whiten,
            resolved_solver,
            backend,
            builder_state,
        }
    }

    /// Principal axes in feature space, with shape `k × d` (one row per
    /// retained component).
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

    /// Per-feature empirical mean learned on the training design.
    pub fn mean(&self) -> ArrayView1<'_, F> {
        self.mean.view()
    }

    /// Concrete solver executed during `fit` (`Full`, `Randomized` or
    /// `CovarianceEigh`; never `Auto`).
    pub fn resolved_solver(&self) -> SKPCASolver {
        self.resolved_solver
    }

    /// Number of retained principal components `k`.
    pub fn number_of_components(&self) -> usize {
        self.components.nrows()
    }

    /// Number of input features `d` seen during `fit`.
    pub fn number_of_features(&self) -> usize {
        self.components.ncols()
    }

    /// Whether `transform` divides projections by the singular-value spread.
    pub fn whiten(&self) -> bool {
        self.whiten
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
