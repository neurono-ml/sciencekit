//! The DBSCAN region query: the index that answers the radius query and the
//! strategy resolution behind it (spec `dbscan`, decision 1).

use sciencekit_common::{SKError, SKFloat};
use sciencekit_neighbors::{SKBallTree, SKBruteForceSearcher, SKKDTree};

use super::config::{SKDBSCANAlgorithm, SKDBSCANMetric, SKDBSCANSearchStrategy};

/// The feature count above which `Auto` prefers a ball tree over a KD-tree.
const BALL_TREE_MINIMUM_FEATURES: usize = 20;

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

/// Resolve the requested algorithm into the strategy the model will use.
///
/// `Auto` builds a tree for dense, tree-compatible data and falls back to the
/// brute-force scan for sparse data and for the cosine distance, which no
/// spatial index can prune. An explicit tree request over sparse data or the
/// cosine metric is rejected with the alternative path named.
pub(crate) fn sk_resolve_strategy(
    algorithm: SKDBSCANAlgorithm,
    metric: SKDBSCANMetric,
    is_dense: bool,
    number_of_features: usize,
) -> Result<SKDBSCANSearchStrategy, SKError> {
    match algorithm {
        SKDBSCANAlgorithm::Auto => Ok(if is_dense && metric.is_tree_compatible() {
            if number_of_features > BALL_TREE_MINIMUM_FEATURES {
                SKDBSCANSearchStrategy::BallTree
            } else {
                SKDBSCANSearchStrategy::KdTree
            }
        } else {
            SKDBSCANSearchStrategy::BruteForce
        }),
        SKDBSCANAlgorithm::BruteForce => Ok(SKDBSCANSearchStrategy::BruteForce),
        SKDBSCANAlgorithm::KdTree | SKDBSCANAlgorithm::BallTree => {
            if !is_dense {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "csr",
                    suggestion: "request SKDBSCANAlgorithm::BruteForce or keep Auto for sparse input",
                });
            }
            if !metric.is_tree_compatible() {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "cosine distance",
                    suggestion: "request SKDBSCANAlgorithm::BruteForce or keep Auto for the cosine metric",
                });
            }
            Ok(match algorithm {
                SKDBSCANAlgorithm::KdTree => SKDBSCANSearchStrategy::KdTree,
                _ => SKDBSCANSearchStrategy::BallTree,
            })
        }
    }
}
