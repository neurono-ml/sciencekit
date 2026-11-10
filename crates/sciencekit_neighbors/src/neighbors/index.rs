//! The optional spatial index a fitted model keeps for the tree strategies
//! (spec `nearest-neighbor-estimators`).

use ndarray::ArrayView2;
use sciencekit_common::{SKBackendKind, SKDataView, SKError, SKExecutionMode, SKFloat};

use super::{SKNeighborsMetric, SKSearchStrategy};
use crate::ball_tree::{SKBallTree, SKBallTreeBuilder};
use crate::kd_tree::{SKKDTree, SKKDTreeBuilder};
use crate::searcher::{SKNearestNeighborsSearcher, SKNeighborQuery};

/// The concrete index behind a tree-backed fitted model.
#[derive(Debug, Clone)]
pub(crate) enum SKNeighborIndex<F> {
    /// A KD-tree index.
    KdTree(SKKDTree<F>),
    /// A ball-tree index.
    BallTree(SKBallTree<F>),
}

impl<F: SKFloat> SKNeighborIndex<F> {
    /// Build the index requested by the resolved strategy.
    pub(crate) fn build(
        strategy: SKSearchStrategy,
        training: ArrayView2<'_, F>,
        leaf_size: usize,
        metric: SKNeighborsMetric,
        backend: SKBackendKind,
        execution_mode: SKExecutionMode,
    ) -> Result<Self, SKError> {
        let tree_metric = metric
            .tree_metric()
            .ok_or(SKError::UnsupportedRepresentation {
                representation: "cosine distance",
                suggestion: "request SKNeighborsAlgorithm::BruteForce for the cosine metric",
            })?;
        match strategy {
            SKSearchStrategy::KdTree => {
                let tree = SKKDTreeBuilder::new()
                    .leaf_size(leaf_size)
                    .metric(tree_metric)
                    .backend(backend)
                    .execution_mode(execution_mode)
                    .build(SKDataView::Dense(training))?;
                Ok(SKNeighborIndex::KdTree(tree))
            }
            SKSearchStrategy::BallTree => {
                let tree = SKBallTreeBuilder::new()
                    .leaf_size(leaf_size)
                    .metric(tree_metric)
                    .backend(backend)
                    .execution_mode(execution_mode)
                    .build(SKDataView::Dense(training))?;
                Ok(SKNeighborIndex::BallTree(tree))
            }
            SKSearchStrategy::BruteForce => Err(SKError::UnsupportedOperation {
                operation: "build_index",
                reason: "the brute-force strategy stores no index".to_string(),
            }),
        }
    }

    /// Query the index with a dense batch.
    pub(crate) fn k_nearest(
        &self,
        queries: ArrayView2<'_, F>,
        neighbors_count: usize,
    ) -> Result<SKNeighborQuery<F>, SKError> {
        match self {
            SKNeighborIndex::KdTree(tree) => {
                tree.k_nearest(SKDataView::Dense(queries), neighbors_count)
            }
            SKNeighborIndex::BallTree(tree) => {
                tree.k_nearest(SKDataView::Dense(queries), neighbors_count)
            }
        }
    }
}
