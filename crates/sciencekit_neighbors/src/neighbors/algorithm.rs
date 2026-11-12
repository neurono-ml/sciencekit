//! Runtime search-strategy selection (spec `nearest-neighbor-estimators`).
//!
//! The estimator carries a requested [`SKNeighborsAlgorithm`]; the search
//! strategy is resolved at fit time from the representation, the metric and the
//! training shape. Resolving once keeps the query path free of repeated
//! branching and makes the decision visible on the fitted model.

use sciencekit_common::SKError;

use super::SKNeighborsMetric;

/// The smallest dense training set for which `Auto` builds a tree.
///
/// A documented engineering choice, not a statistical claim: below it the tree
/// construction costs more than scanning the rows.
pub const SK_AUTO_TREE_MINIMUM_TRAINING_ROWS: usize = 64;

/// The feature count above which `Auto` prefers a ball tree over a KD-tree.
pub const SK_AUTO_BALL_TREE_MINIMUM_FEATURES: usize = 20;

/// The search algorithm a caller can request from a builder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum SKNeighborsAlgorithm {
    /// Pick the strategy from the representation, metric and training shape.
    #[default]
    Auto,
    /// A KD-tree index (dense, tree-compatible metrics only).
    KdTree,
    /// A ball-tree index (dense, tree-compatible metrics only).
    BallTree,
    /// A direct scan over the stored training design.
    BruteForce,
}

/// The input representation the strategy is resolved for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SKFeatureRepresentation {
    /// A row-major dense matrix, eligible for tree geometry.
    Dense,
    /// A compressed sparse row matrix, served by brute force only.
    Sparse,
}

/// The concrete search strategy of a fitted model. Never `Auto`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SKSearchStrategy {
    /// Exact search through a KD-tree.
    KdTree,
    /// Exact search through a ball tree.
    BallTree,
    /// Exact search by scanning the stored training rows.
    BruteForce,
}

impl SKSearchStrategy {
    /// Whether the strategy is backed by a tree index.
    pub fn is_tree(self) -> bool {
        matches!(self, SKSearchStrategy::KdTree | SKSearchStrategy::BallTree)
    }
}

/// Resolve the requested algorithm into the strategy a fitted model will use.
///
/// `Auto` builds a tree only for dense input under a tree-compatible metric
/// with at least [`SK_AUTO_TREE_MINIMUM_TRAINING_ROWS`] rows; sparse input and
/// cosine distance fall back to brute force. An explicit tree request that the
/// representation or the metric cannot serve returns
/// [`SKError::UnsupportedRepresentation`] with the alternative path named.
pub fn sk_resolve_search_strategy(
    algorithm: SKNeighborsAlgorithm,
    metric: SKNeighborsMetric,
    representation: SKFeatureRepresentation,
    training_rows: usize,
    training_columns: usize,
) -> Result<SKSearchStrategy, SKError> {
    let dense = representation == SKFeatureRepresentation::Dense;
    let tree_compatible = dense && metric.is_tree_compatible();
    match algorithm {
        SKNeighborsAlgorithm::Auto => Ok(sk_auto_strategy(
            tree_compatible,
            training_rows,
            training_columns,
        )),
        SKNeighborsAlgorithm::BruteForce => Ok(SKSearchStrategy::BruteForce),
        SKNeighborsAlgorithm::KdTree | SKNeighborsAlgorithm::BallTree => {
            if !dense {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "csr",
                    suggestion: "request SKNeighborsAlgorithm::BruteForce or keep Auto for sparse input",
                });
            }
            if !metric.is_tree_compatible() {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "cosine distance",
                    suggestion: "request SKNeighborsAlgorithm::BruteForce for the cosine metric",
                });
            }
            Ok(match algorithm {
                SKNeighborsAlgorithm::KdTree => SKSearchStrategy::KdTree,
                _ => SKSearchStrategy::BallTree,
            })
        }
    }
}

/// The documented `Auto` policy for a tree-eligible input.
fn sk_auto_strategy(
    tree_compatible: bool,
    training_rows: usize,
    training_columns: usize,
) -> SKSearchStrategy {
    if !tree_compatible || training_rows < SK_AUTO_TREE_MINIMUM_TRAINING_ROWS {
        return SKSearchStrategy::BruteForce;
    }
    if training_columns > SK_AUTO_BALL_TREE_MINIMUM_FEATURES {
        SKSearchStrategy::BallTree
    } else {
        SKSearchStrategy::KdTree
    }
}
