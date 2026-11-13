//! The searcher type implementing the shared contract over any representation.

use sciencekit_common::{SKDataView, SKError, SKFloat};

use super::super::{SKNeighborsMetric, SKTrainingData, sk_brute_force_k_nearest};
use crate::searcher::{SKNearestNeighborsSearcher, SKNeighborQuery, SKRadiusNeighbors};

/// An exact searcher that scans the stored training rows.
///
/// It serves every representation the pairwise dispatcher understands — dense,
/// CSR and any dense/sparse mix — without densifying sparse input. Distances
/// come from [`sciencekit_math::pairwise::sk_pairwise_distance_matrix`], in
/// chunks sized from a fixed memory budget, and ties are broken by the smallest
/// training row index.
#[derive(Debug, Clone)]
pub struct SKBruteForceSearcher<F> {
    training: SKTrainingData<F>,
    metric: SKNeighborsMetric,
}

impl<F: SKFloat> SKBruteForceSearcher<F> {
    /// Build the searcher from an owned training design.
    ///
    /// Rejects an empty design and any non-finite training value before a
    /// query can run.
    pub fn build(training: SKTrainingData<F>, metric: SKNeighborsMetric) -> Result<Self, SKError> {
        if training.rows() == 0 || training.columns() == 0 {
            return Err(SKError::EmptyInput { operation: "build" });
        }
        training.validate_finite()?;
        Ok(SKBruteForceSearcher { training, metric })
    }

    /// The number of training rows the searcher holds.
    pub fn training_rows(&self) -> usize {
        self.training.rows()
    }

    /// The number of features per training row.
    pub fn training_features(&self) -> usize {
        self.training.columns()
    }

    /// The configured distance.
    pub fn metric(&self) -> SKNeighborsMetric {
        self.metric
    }

    /// The borrowed training design.
    pub fn training_data(&self) -> &SKTrainingData<F> {
        &self.training
    }
}

impl<F: SKFloat> SKNearestNeighborsSearcher<F> for SKBruteForceSearcher<F> {
    fn k_nearest(
        &self,
        queries: SKDataView<'_, F>,
        neighbors_count: usize,
    ) -> Result<SKNeighborQuery<F>, SKError> {
        sk_brute_force_k_nearest(self.training.view(), self.metric, queries, neighbors_count)
    }

    fn radius_neighbors(
        &self,
        queries: SKDataView<'_, F>,
        radius: F,
    ) -> Result<SKRadiusNeighbors<F>, SKError> {
        super::radius::sk_brute_force_radius_neighbors(
            self.training.view(),
            self.metric,
            queries,
            radius,
        )
    }
}
