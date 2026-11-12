//! Incoming-edge resolution and edge-order horizontal block concatenation for
//! [`super::SKPipelineGraph`] (spec `pipeline-graph`, Task 2.4).
//!
//! # Merge Policy
//!
//! - **Zero incoming edges (root node):** receives the caller's input
//!   [`SKFeatureBlock`].
//! - **One incoming edge:** receives that predecessor's output block without
//!   copying or densifying.
//! - **Multiple incoming edges (merge):** incoming blocks are concatenated
//!   horizontally along the column axis in **exact `connect(source, target)`
//!   edge insertion order** (never in node-identifier order or `petgraph`'s
//!   internal reverse adjacency order). All-dense blocks stay dense; if any
//!   incoming block is sparse CSR, the merged block is CSR without densifying
//!   sparse inputs.

use petgraph::Direction;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use sciencekit_common::{SKError, SKFeatureBlock, SKFloat};

use crate::feature_union::sk_concatenate_blocks;

use super::errors::SKPipelineGraphNodeError;
use super::node::{GraphNodePayload, SKNodeIdentifier};
use super::order::SKNodeBlockStore;

/// Predecessor node identifiers feeding `node`, ordered by ascending
/// `connect(source, target)` insertion order.
pub(super) fn sk_ordered_predecessors<F: SKFloat>(
    graph: &DiGraph<GraphNodePayload<F>, usize>,
    node: SKNodeIdentifier,
) -> Vec<SKNodeIdentifier> {
    if node.index() >= graph.node_count() {
        return Vec::new();
    }
    let mut incoming_edges: Vec<(usize, SKNodeIdentifier)> = graph
        .edges_directed(NodeIndex::new(node.index()), Direction::Incoming)
        .map(|edge| (*edge.weight(), SKNodeIdentifier::new(edge.source().index())))
        .collect();
    incoming_edges.sort_by_key(|&(insertion_order, _)| insertion_order);
    incoming_edges
        .into_iter()
        .map(|(_, predecessor)| predecessor)
        .collect()
}

/// Resolve the incoming [`SKFeatureBlock`] for `node` according to the merge
/// policy.
pub(super) fn sk_resolve_incoming_block<F: SKFloat>(
    graph: &DiGraph<GraphNodePayload<F>, usize>,
    node: SKNodeIdentifier,
    input_block: &SKFeatureBlock<F>,
    block_store: &SKNodeBlockStore<F>,
    expected_widths: Option<&[usize]>,
) -> Result<(SKFeatureBlock<F>, Vec<usize>), SKPipelineGraphNodeError> {
    let predecessors = sk_ordered_predecessors(graph, node);
    if predecessors.is_empty() {
        let width = input_block.number_of_columns();
        return Ok((input_block.clone(), vec![width]));
    }

    let blocks: Vec<SKFeatureBlock<F>> = predecessors
        .iter()
        .map(|&predecessor| {
            block_store.get(predecessor).cloned().ok_or_else(|| {
                SKPipelineGraphNodeError::node(
                    node,
                    SKError::Conversion(format!("missing output block from {predecessor}")),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    if blocks.len() == 1 {
        let single = blocks.into_iter().next().unwrap();
        let width = single.number_of_columns();
        if let Some(&[expected_width]) = expected_widths {
            if width != expected_width {
                let rows = input_block.number_of_rows();
                return Err(SKPipelineGraphNodeError::node(
                    node,
                    SKError::ShapeMismatch {
                        expected: vec![rows, expected_width],
                        found: vec![rows, width],
                    },
                ));
            }
        }
        return Ok((single, vec![width]));
    }

    let widths: Vec<usize> = match expected_widths {
        Some(recorded) => recorded.to_vec(),
        None => blocks
            .iter()
            .map(SKFeatureBlock::number_of_columns)
            .collect(),
    };
    let total_width: usize = widths.iter().copied().sum();
    let merged = sk_concatenate_blocks(&blocks, input_block.number_of_rows(), &widths, total_width)
        .map_err(|union_error| SKPipelineGraphNodeError::node(node, union_error.into_sk_error()))?;
    Ok((merged, widths))
}
