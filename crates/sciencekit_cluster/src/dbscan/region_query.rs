//! The DBSCAN region query: the index that answers the radius query and the
//! strategy resolution behind it (spec `dbscan`, decision 1).

use sciencekit_common::SKFloat;
use sciencekit_neighbors::{SKBallTree, SKBruteForceSearcher, SKKDTree};

/// The concrete spatial index behind a DBSCAN region query.
///
/// The variant is resolved once, at fit time, from the requested algorithm, the
/// representation and the metric. It is never re-resolved per operation.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) enum SKDBSCANIndex<F: SKFloat> {
    /// A KD-tree index over dense rows.
    KdTree(SKKDTree<F>),
    /// A ball-tree index over dense rows.
    BallTree(SKBallTree<F>),
    /// A direct scan over the stored rows, in any representation.
    BruteForce(SKBruteForceSearcher<F>),
}
