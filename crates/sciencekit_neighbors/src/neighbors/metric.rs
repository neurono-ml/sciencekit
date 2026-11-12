//! The distance catalogue offered by the neighbor estimators
//! (spec `nearest-neighbor-estimators`).

use sciencekit_math::pairwise::SKPairwiseMetric;

use crate::metric::SKNeighborMetric;

/// The distance a neighbor estimator measures.
///
/// The catalogue is the estimator surface: every variant is accepted on dense
/// and sparse input through the pairwise dispatcher. Only the first three are
/// compatible with the spatial indexes, because tree pruning requires a
/// distance obeying the triangle inequality; [`Self::Cosine`] is served by the
/// brute-force searcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SKNeighborsMetric {
    /// The Euclidean (`L2`) distance.
    #[default]
    Euclidean,
    /// The Manhattan (`L1`) distance.
    Manhattan,
    /// The Chebyshev (`L∞`) distance.
    Chebyshev,
    /// The cosine distance, `1 − cos θ`.
    Cosine,
}

impl SKNeighborsMetric {
    /// Every metric the estimators accept, in catalogue order.
    pub const ALL: [SKNeighborsMetric; 4] = [
        SKNeighborsMetric::Euclidean,
        SKNeighborsMetric::Manhattan,
        SKNeighborsMetric::Chebyshev,
        SKNeighborsMetric::Cosine,
    ];

    /// The pairwise kernel selector used by the brute-force searcher.
    pub fn pairwise_metric(self) -> SKPairwiseMetric {
        match self {
            SKNeighborsMetric::Euclidean => SKPairwiseMetric::Euclidean,
            SKNeighborsMetric::Manhattan => SKPairwiseMetric::Manhattan,
            SKNeighborsMetric::Chebyshev => SKPairwiseMetric::Chebyshev,
            SKNeighborsMetric::Cosine => SKPairwiseMetric::Cosine,
        }
    }

    /// The equivalent spatial-index metric, or `None` when the distance cannot
    /// prune a tree.
    pub fn tree_metric(self) -> Option<SKNeighborMetric> {
        match self {
            SKNeighborsMetric::Euclidean => Some(SKNeighborMetric::Euclidean),
            SKNeighborsMetric::Manhattan => Some(SKNeighborMetric::Manhattan),
            SKNeighborsMetric::Chebyshev => Some(SKNeighborMetric::Chebyshev),
            SKNeighborsMetric::Cosine => None,
        }
    }

    /// Whether a spatial index can serve this distance.
    pub fn is_tree_compatible(self) -> bool {
        self.tree_metric().is_some()
    }
}
