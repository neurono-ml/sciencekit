//! Deterministic level-ordered topological sorting and per-node feature block
//! storage (spec `pipeline-graph`, Task 2.1).

use std::collections::BTreeMap;

use petgraph::Direction;
use petgraph::algo::toposort;
use petgraph::graph::DiGraph;
use sciencekit_common::{SKError, SKFeatureBlock, SKFloat};

use super::node::{GraphNodePayload, SKNodeIdentifier};

/// Deterministic map of [`SKFeatureBlock`] outputs keyed by [`SKNodeIdentifier`].
#[derive(Debug, Clone)]
pub struct SKNodeBlockStore<F: SKFloat> {
    blocks: BTreeMap<SKNodeIdentifier, SKFeatureBlock<F>>,
}

impl<F: SKFloat> SKNodeBlockStore<F> {
    /// Construct an empty per-node block store.
    #[allow(dead_code)]
    pub(crate) fn new() -> Self {
        Self {
            blocks: BTreeMap::new(),
        }
    }

    /// Record `block` as the output associated with `node`.
    #[allow(dead_code)]
    pub(crate) fn insert(&mut self, node: SKNodeIdentifier, block: SKFeatureBlock<F>) {
        self.blocks.insert(node, block);
    }

    /// Borrow the block stored for `node`, if present.
    pub fn get(&self, node: SKNodeIdentifier) -> Option<&SKFeatureBlock<F>> {
        self.blocks.get(&node)
    }

    /// Number of node blocks stored.
    pub fn len(&self) -> usize {
        self.blocks.len()
    }

    /// Whether no blocks are stored.
    pub fn is_empty(&self) -> bool {
        self.blocks.is_empty()
    }

    /// Borrow the underlying ordered map indexed by [`SKNodeIdentifier`].
    pub fn as_map(&self) -> &BTreeMap<SKNodeIdentifier, SKFeatureBlock<F>> {
        &self.blocks
    }
}

/// Compute the deterministic topological execution order: nodes are grouped by
/// topological level (`0` for input roots, `1 + max(predecessor levels)`
/// otherwise) and ordered by ascending [`SKNodeIdentifier`] within each level.
pub(super) fn sk_topological_level_order<F: SKFloat>(
    graph: &DiGraph<GraphNodePayload<F>, usize>,
) -> Result<Vec<SKNodeIdentifier>, SKError> {
    let raw_order = toposort(graph, None).map_err(|cycle| SKError::InvalidHyperparameter {
        name: "edges",
        reason: format!(
            "pipeline graph contains a cycle involving {}",
            SKNodeIdentifier::new(cycle.node_id().index())
        ),
    })?;

    let mut levels = BTreeMap::<usize, usize>::new();
    raw_order.iter().for_each(|&node| {
        let level = graph
            .neighbors_directed(node, Direction::Incoming)
            .filter_map(|predecessor| levels.get(&predecessor.index()).map(|depth| depth + 1))
            .max()
            .unwrap_or(0);
        levels.insert(node.index(), level);
    });

    let mut ranked: Vec<(usize, SKNodeIdentifier)> = raw_order
        .into_iter()
        .map(|node| {
            let level = levels.get(&node.index()).copied().unwrap_or(0);
            (level, SKNodeIdentifier::new(node.index()))
        })
        .collect();
    ranked.sort_by_key(|&(level, identifier)| (level, identifier));

    Ok(ranked
        .into_iter()
        .map(|(_, identifier)| identifier)
        .collect())
}
