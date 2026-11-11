//! Incoming-edge resolution and edge-order horizontal block concatenation for
//! [`super::SKPipelineGraph`] (spec `pipeline-graph`).

use petgraph::Direction;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::visit::EdgeRef;
use sciencekit_common::{SKError, SKFeatureBlock, SKFloat};

use crate::feature_union::sk_concatenate_blocks;

use super::errors::SKPipelineGraphNodeError;
use super::node::{GraphNodePayload, SKNodeIdentifier};
use super::order::SKNodeBlockStore;

/// Resolve the incoming [`SKFeatureBlock`] for `node`:
/// - `0` incoming edges: root node receives `input_block`.
/// - `1` incoming edge: receives the predecessor's output block.
/// - `>= 2` incoming edges: concatenates predecessor blocks horizontally in
///   exact `connect(source, target)` edge insertion order.
pub(super) fn sk_resolve_incoming_block<F: SKFloat>(
    graph: &DiGraph<GraphNodePayload<F>, usize>,
    node: SKNodeIdentifier,
    input_block: &SKFeatureBlock<F>,
    block_store: &SKNodeBlockStore<F>,
    expected_widths: Option<&[usize]>,
) -> Result<(SKFeatureBlock<F>, Vec<usize>), SKPipelineGraphNodeError> {
    let mut incoming_edges: Vec<(usize, SKNodeIdentifier)> = graph
        .edges_directed(NodeIndex::new(node.index()), Direction::Incoming)
        .map(|edge| (*edge.weight(), SKNodeIdentifier::new(edge.source().index())))
        .collect();
    incoming_edges.sort_by_key(|&(insertion_order, _)| insertion_order);

    if incoming_edges.is_empty() {
        let width = input_block.number_of_columns();
        return Ok((input_block.clone(), vec![width]));
    }

    let blocks: Vec<SKFeatureBlock<F>> = incoming_edges
        .iter()
        .map(|&(_, predecessor)| {
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
