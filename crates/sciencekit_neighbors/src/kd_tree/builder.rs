//! The builder-only construction surface for [`SKKDTree`].

use sciencekit_common::{
    SKBackendKind, SKDataView, SKError, SKExecutionMode, SKFloat,
    builders::{SKBuilderState, sk_validate_hyperparameter},
};

use super::build::SKKDTree;
use crate::metric::SKRequestedNeighborMetric;

/// Builder for an immutable [`SKKDTree`].
///
/// Construction is builder-only; the direct constructor stays private. The
/// execution intent defaults to [`SKExecutionMode::Automatic`] and the backend to
/// [`SKBackendKind::Faer`].
pub struct SKKDTreeBuilder<F> {
    state: SKBuilderState,
    leaf_size: usize,
    requested_metric: SKRequestedNeighborMetric,
    backend: SKBackendKind,
    marker: std::marker::PhantomData<F>,
}

impl<F: SKFloat> SKKDTreeBuilder<F> {
    /// A builder with the documented defaults: `leaf_size = 40`, Euclidean
    /// metric, `Faer` backend and automatic execution mode.
    pub fn new() -> SKKDTreeBuilder<F> {
        SKKDTreeBuilder {
            state: SKBuilderState::new(),
            leaf_size: 40,
            requested_metric: SKRequestedNeighborMetric::Euclidean,
            backend: SKBackendKind::Faer,
            marker: std::marker::PhantomData,
        }
    }

    /// The maximum number of rows in a leaf; defaults to `40`.
    pub fn leaf_size(mut self, leaf_size: usize) -> Self {
        self.leaf_size = leaf_size;
        self
    }

    /// The distance metric; defaults to Euclidean. Cosine and NaN-aware
    /// Euclidean are accepted by the type but rejected at [`Self::build`].
    pub fn metric(mut self, metric: impl Into<SKRequestedNeighborMetric>) -> Self {
        self.requested_metric = metric.into();
        self
    }

    /// The backend kind recorded on the build span; defaults to
    /// [`SKBackendKind::Faer`].
    pub fn backend(mut self, backend: SKBackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Override the execution intent; defaults to
    /// [`SKExecutionMode::Automatic`].
    pub fn execution_mode<E>(mut self, mode: E) -> Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    /// Build the immutable index from a borrowed dense training view.
    pub fn build(self, features: SKDataView<'_, F>) -> Result<SKKDTree<F>, SKError> {
        let dense = match features {
            SKDataView::Dense(view) => view,
            _ => {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "csr",
                    suggestion: "use the brute-force searcher for sparse input, delivered by the knn-classifier-regressor change",
                });
            }
        };

        if dense.nrows() == 0 || dense.ncols() == 0 {
            return Err(SKError::EmptyInput { operation: "build" });
        }
        sk_validate_hyperparameter(
            "leaf_size",
            self.leaf_size > 0,
            "leaf_size must be strictly positive",
        )?;
        let metric = match self.requested_metric.supported() {
            Some(metric) => metric,
            None => {
                return Err(SKError::InvalidHyperparameter {
                    name: "metric",
                    reason: format!(
                        "{:?} is not supported by the spatial index: tree pruning requires a distance obeying the triangle inequality",
                        self.requested_metric
                    ),
                });
            }
        };
        for (position, value) in dense.iter().enumerate() {
            if !value.is_finite() {
                return Err(SKError::NonFiniteValue { position });
            }
        }

        let span = tracing::info_span!(
            "sciencekit.neighbors.build",
            operation = "build",
            backend = %self.backend,
            rows = dense.nrows(),
            columns = dense.ncols(),
        );
        let _entered = span.enter();
        Ok(SKKDTree::build_index(
            dense.to_owned(),
            self.leaf_size,
            metric,
            self.backend,
            self.state.execution_intent(),
        ))
    }
}

impl<F: SKFloat> Default for SKKDTreeBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}
