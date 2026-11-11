//! The DBSCAN region query: the index that answers the radius query and the
//! strategy resolution behind it (spec `dbscan`, decision 1).

use sciencekit_common::{SKDataView, SKError, SKFloat};
use sciencekit_neighbors::{
    SKBallTree, SKBallTreeBuilder, SKBruteForceSearcher, SKKDTree, SKKDTreeBuilder,
    SKNearestNeighborsSearcher, SKNeighborMetric, SKNeighborQuery, SKRadiusNeighbors,
    SKTrainingData,
};

use super::config::{
    SKDBSCANAlgorithm, SKDBSCANConfiguration, SKDBSCANMetric, SKDBSCANSearchStrategy,
};

/// The feature count above which `Auto` prefers a ball tree over a KD-tree.
const BALL_TREE_MINIMUM_FEATURES: usize = 20;

/// The concrete spatial index behind a DBSCAN region query.
///
/// The variant is resolved once, at fit time, from the requested algorithm, the
/// representation and the metric. It is never re-resolved per operation.
#[derive(Debug, Clone)]
pub(crate) enum SKDBSCANIndex<F: SKFloat> {
    /// A KD-tree index over dense rows.
    KdTree(SKKDTree<F>),
    /// A ball-tree index over dense rows.
    BallTree(SKBallTree<F>),
    /// A direct scan over the stored rows, in any representation.
    BruteForce(SKBruteForceSearcher<F>),
}

impl<F: SKFloat> SKDBSCANIndex<F> {
    /// Every indexed row within the inclusive `radius` of every query row.
    pub(crate) fn radius_neighbors(
        &self,
        queries: SKDataView<'_, F>,
        radius: F,
    ) -> Result<SKRadiusNeighbors<F>, SKError> {
        match self {
            Self::KdTree(tree) => tree.radius_neighbors(queries, radius),
            Self::BallTree(tree) => tree.radius_neighbors(queries, radius),
            Self::BruteForce(searcher) => searcher.radius_neighbors(queries, radius),
        }
    }

    /// Exact `k`-nearest indexed rows for every query row.
    pub(crate) fn k_nearest(
        &self,
        queries: SKDataView<'_, F>,
        neighbors_count: usize,
    ) -> Result<SKNeighborQuery<F>, SKError> {
        match self {
            Self::KdTree(tree) => tree.k_nearest(queries, neighbors_count),
            Self::BallTree(tree) => tree.k_nearest(queries, neighbors_count),
            Self::BruteForce(searcher) => searcher.k_nearest(queries, neighbors_count),
        }
    }
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

/// Build the index answering the DBSCAN radius query over `training`.
pub(crate) fn sk_build_index<F: SKFloat>(
    configuration: &SKDBSCANConfiguration<F>,
    training: &SKTrainingData<F>,
    strategy: SKDBSCANSearchStrategy,
) -> Result<SKDBSCANIndex<F>, SKError> {
    match strategy {
        SKDBSCANSearchStrategy::KdTree => {
            let dense = training.dense().ok_or(SKError::UnsupportedRepresentation {
                representation: "csr",
                suggestion: "request SKDBSCANAlgorithm::BruteForce for sparse input",
            })?;
            let tree = SKKDTreeBuilder::new()
                .leaf_size(configuration.leaf_size)
                .metric(sk_tree_metric(configuration.metric)?)
                .backend(configuration.backend)
                .execution_mode(configuration.execution_mode)
                .build(SKDataView::Dense(dense))?;
            Ok(SKDBSCANIndex::KdTree(tree))
        }
        SKDBSCANSearchStrategy::BallTree => {
            let dense = training.dense().ok_or(SKError::UnsupportedRepresentation {
                representation: "csr",
                suggestion: "request SKDBSCANAlgorithm::BruteForce for sparse input",
            })?;
            let tree = SKBallTreeBuilder::new()
                .leaf_size(configuration.leaf_size)
                .metric(sk_tree_metric(configuration.metric)?)
                .backend(configuration.backend)
                .execution_mode(configuration.execution_mode)
                .build(SKDataView::Dense(dense))?;
            Ok(SKDBSCANIndex::BallTree(tree))
        }
        SKDBSCANSearchStrategy::BruteForce => {
            let searcher = SKBruteForceSearcher::build(
                training.clone(),
                configuration.metric.neighbors_metric(),
            )?;
            Ok(SKDBSCANIndex::BruteForce(searcher))
        }
    }
}

/// Every training row within the inclusive `eps` of every training row, sorted
/// in ascending training-index order and always containing the row itself.
pub(crate) fn sk_radius_neighborhoods<F: SKFloat>(
    index: &SKDBSCANIndex<F>,
    training: &SKTrainingData<F>,
    eps: F,
) -> Result<Vec<Vec<usize>>, SKError> {
    let result = index.radius_neighbors(training.view(), eps)?;
    Ok((0..result.rows())
        .map(|row| {
            let mut neighbors = result.indices(row).to_vec();
            neighbors.sort_unstable();
            if let Err(insert_at) = neighbors.binary_search(&row) {
                neighbors.insert(insert_at, row);
            }
            neighbors
        })
        .collect())
}

/// The spatial-index metric of a DBSCAN metric, or the documented rejection.
fn sk_tree_metric(metric: SKDBSCANMetric) -> Result<SKNeighborMetric, SKError> {
    metric
        .neighbors_metric()
        .tree_metric()
        .ok_or(SKError::UnsupportedRepresentation {
            representation: "cosine distance",
            suggestion: "request SKDBSCANAlgorithm::BruteForce or keep Auto for the cosine metric",
        })
}
