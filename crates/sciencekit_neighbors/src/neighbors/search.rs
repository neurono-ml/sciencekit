//! Fit-time search preparation and query dispatch shared by both estimators
//! (spec `nearest-neighbor-estimators`).

use sciencekit_common::{SKDataView, SKError, SKFloat};

use super::index::SKNeighborIndex;
use super::validation::{
    sk_validate_neighbors_count, sk_validate_training_features, sk_view_dimensions,
};
use super::{
    SKFeatureRepresentation, SKNeighborsConfiguration, SKNeighborsMetric, SKSearchStrategy,
    SKTrainingData, sk_brute_force_k_nearest, sk_resolve_search_strategy,
};
use crate::searcher::SKNeighborQuery;

/// The fitted search state: the owned training design, the optional index and
/// the resolved strategy.
#[derive(Debug, Clone)]
pub(crate) struct SKFittedSearch<F> {
    /// The training design, in its original representation.
    pub(crate) training: SKTrainingData<F>,
    /// The tree index, present only for a tree strategy.
    pub(crate) index: Option<SKNeighborIndex<F>>,
    /// The strategy resolved at fit time (never `Auto`).
    pub(crate) strategy: SKSearchStrategy,
}

impl<F: SKFloat> SKFittedSearch<F> {
    /// The number of training rows.
    pub(crate) fn training_rows(&self) -> usize {
        self.training.rows()
    }

    /// The number of features per training row.
    pub(crate) fn number_of_features(&self) -> usize {
        self.training.columns()
    }

    /// Query the fitted search, falling back to brute force whenever the query
    /// representation or the strategy cannot use the stored index.
    pub(crate) fn k_nearest(
        &self,
        metric: SKNeighborsMetric,
        queries: SKDataView<'_, F>,
        neighbors_count: usize,
    ) -> Result<SKNeighborQuery<F>, SKError> {
        match (&self.index, queries) {
            (Some(index), SKDataView::Dense(dense)) => index.k_nearest(dense, neighbors_count),
            _ => sk_brute_force_k_nearest(self.training.view(), metric, queries, neighbors_count),
        }
    }
}

/// Validate a training input and build the search state for it.
pub(crate) fn sk_prepare_search<F: SKFloat>(
    features: SKDataView<'_, F>,
    configuration: &SKNeighborsConfiguration,
) -> Result<SKFittedSearch<F>, SKError> {
    sk_validate_training_features(features, "fit")?;
    let (training_rows, training_columns) = sk_view_dimensions(&features);
    sk_validate_neighbors_count(configuration.nearest_neighbors_count, training_rows, "fit")?;
    let representation = match features {
        SKDataView::Dense(_) => SKFeatureRepresentation::Dense,
        SKDataView::Sparse(_) => SKFeatureRepresentation::Sparse,
        _ => {
            return Err(SKError::UnsupportedRepresentation {
                representation: features.representation(),
                suggestion: "convert the input to a dense matrix or a CSR matrix",
            });
        }
    };
    let strategy = sk_resolve_search_strategy(
        configuration.algorithm,
        configuration.metric,
        representation,
        training_rows,
        training_columns,
    )?;
    let training = SKTrainingData::from_view(features);
    let index = if strategy.is_tree() {
        let dense = training.dense().ok_or(SKError::UnsupportedRepresentation {
            representation: "csr",
            suggestion: "request SKNeighborsAlgorithm::BruteForce for sparse input",
        })?;
        Some(SKNeighborIndex::build(
            strategy,
            dense,
            configuration.leaf_size,
            configuration.metric,
            configuration.backend,
            configuration.execution_mode,
        )?)
    } else {
        None
    };
    Ok(SKFittedSearch {
        training,
        index,
        strategy,
    })
}
