//! The DBSCAN configuration vocabulary and the validated configuration
//! (spec `dbscan`, decision 6).

use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};
use sciencekit_neighbors::SKNeighborsMetric;

/// The distance a DBSCAN neighbourhood is measured with.
///
/// The catalogue is closed. Euclidean is the default and is served by every
/// search strategy; Manhattan and Chebyshev are served by every strategy as
/// well; the cosine distance is served by the brute-force scan only, because a
/// spatial index cannot prune a distance violating the triangle inequality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SKDBSCANMetric {
    /// The Euclidean (`L2`) distance.
    #[default]
    Euclidean,
    /// The Manhattan (`L1`) distance.
    Manhattan,
    /// The Chebyshev (`L∞`) distance.
    Chebyshev,
    /// The cosine distance, `1 − cos θ`; served by the brute-force scan.
    Cosine,
}

impl SKDBSCANMetric {
    /// Every metric the estimator accepts, in catalogue order.
    pub const ALL: [SKDBSCANMetric; 4] = [
        SKDBSCANMetric::Euclidean,
        SKDBSCANMetric::Manhattan,
        SKDBSCANMetric::Chebyshev,
        SKDBSCANMetric::Cosine,
    ];

    /// Whether a spatial index can serve this distance.
    ///
    /// The cosine distance is excluded: tree pruning requires a distance
    /// obeying the triangle inequality.
    pub fn is_tree_compatible(self) -> bool {
        !matches!(self, SKDBSCANMetric::Cosine)
    }

    /// Maps the metric onto the shared `sciencekit_neighbors` metric catalogue.
    pub(crate) fn neighbors_metric(self) -> SKNeighborsMetric {
        match self {
            SKDBSCANMetric::Euclidean => SKNeighborsMetric::Euclidean,
            SKDBSCANMetric::Manhattan => SKNeighborsMetric::Manhattan,
            SKDBSCANMetric::Chebyshev => SKNeighborsMetric::Chebyshev,
            SKDBSCANMetric::Cosine => SKNeighborsMetric::Cosine,
        }
    }
}

/// The neighbourhood-search algorithm a caller can request.
///
/// `Auto` resolves the strategy from the representation, the metric and the
/// training shape at fit time; `KdTree` and `BallTree` request a dense
/// spatial index; `BruteForce` requests a direct scan over the stored rows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SKDBSCANAlgorithm {
    /// Pick the strategy from the representation, the metric and the shape.
    #[default]
    Auto,
    /// A KD-tree index (dense, tree-compatible metrics only).
    KdTree,
    /// A ball-tree index (dense, tree-compatible metrics only).
    BallTree,
    /// A direct scan over the stored training design.
    BruteForce,
}

/// The search strategy a fitted model resolved at fit time.
///
/// The value is never `Auto`: the resolution happens once, at fit time, and the
/// fitted model reports the strategy it actually used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SKDBSCANSearchStrategy {
    /// An exact search through a KD-tree.
    KdTree,
    /// An exact search through a ball tree.
    BallTree,
    /// An exact search by scanning the stored training rows.
    BruteForce,
}

impl SKDBSCANSearchStrategy {
    /// Whether the strategy is backed by a spatial index.
    pub fn is_tree(self) -> bool {
        matches!(
            self,
            SKDBSCANSearchStrategy::KdTree | SKDBSCANSearchStrategy::BallTree
        )
    }
}

/// The validated configuration carried by a built estimator.
///
/// The values are checked by the builder before an estimator exists, so every
/// estimator in circulation holds a configuration that can be fitted.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SKDBSCANConfiguration<F: SKFloat> {
    /// The neighbourhood radius, strictly positive and finite.
    pub(crate) eps: F,
    /// The number of points (the point itself included) that make a core point.
    pub(crate) minimum_samples: usize,
    /// The distance the neighbourhood is measured with.
    pub(crate) metric: SKDBSCANMetric,
    /// The requested neighbourhood-search algorithm.
    pub(crate) algorithm: SKDBSCANAlgorithm,
    /// The maximum number of rows in a spatial-index leaf.
    pub(crate) leaf_size: usize,
    /// Backend kind recorded in the operation spans.
    pub(crate) backend: SKBackendKind,
    /// Execution intent.
    pub(crate) execution_mode: SKExecutionMode,
}
