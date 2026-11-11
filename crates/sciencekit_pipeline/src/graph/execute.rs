//! Topological execution of [`super::SKPipelineGraph`] during fitting and
//! inference (spec `pipeline-graph`).

use std::collections::BTreeMap;

use petgraph::graph::{DiGraph, NodeIndex};
use sciencekit_common::{SKError, SKFeatureBlock, SKFloat, SKTargetView};

use crate::SKBoxedFittedStage;

use super::errors::{SKPipelineGraphNodeError, sk_validate_node_block};
use super::final_estimator::SKBoxedFinalModel;
use super::merge::sk_resolve_incoming_block;
use super::node::{GraphNodePayload, SKNodeIdentifier};
use super::order::SKNodeBlockStore;

pub(super) struct FittedGraphExecution<F: SKFloat> {
    pub(super) fitted_stages: BTreeMap<SKNodeIdentifier, SKBoxedFittedStage>,
    pub(super) incoming_widths: BTreeMap<SKNodeIdentifier, Vec<usize>>,
    pub(super) fitted_node_blocks: SKNodeBlockStore<F>,
    pub(super) final_model: SKBoxedFinalModel<F>,
    pub(super) number_of_features: usize,
}

/// Fit every node of `graph` in deterministic `topological_order`.
pub(super) fn sk_fit_graph<F: SKFloat>(
    graph: &DiGraph<GraphNodePayload<F>, usize>,
    topological_order: &[SKNodeIdentifier],
    input_block: &SKFeatureBlock<F>,
    targets: SKTargetView<'_, F>,
) -> Result<FittedGraphExecution<F>, SKPipelineGraphNodeError> {
    let expected_rows = input_block.number_of_rows();
    sk_validate_node_block(None, input_block, expected_rows)?;

    let mut fitted_stages = BTreeMap::new();
    let mut incoming_widths = BTreeMap::new();
    let mut fitted_node_blocks = SKNodeBlockStore::new();
    let mut final_outcome: Option<(SKBoxedFinalModel<F>, usize)> = None;

    topological_order.iter().try_for_each(|&node| {
        let (incoming_block, widths) =
            sk_resolve_incoming_block(graph, node, input_block, &fitted_node_blocks, None)?;
        incoming_widths.insert(node, widths);

        let payload = &graph[NodeIndex::new(node.index())];
        let span = tracing::info_span!(
            "pipeline_graph_node",
            node_identifier = node.index(),
            node_kind = payload.kind().label(),
            operation = "fit"
        );
        let _guard = span.enter();

        match payload {
            GraphNodePayload::Transformer(stage) | GraphNodePayload::FeatureUnion(stage) => {
                let fitted = stage
                    .fit_block(&incoming_block)
                    .map_err(|error| SKPipelineGraphNodeError::node(node, error))?;
                let output = stage
                    .transform_block(&fitted, &incoming_block)
                    .map_err(|error| SKPipelineGraphNodeError::node(node, error))?;
                sk_validate_node_block(Some(node), &output, expected_rows)?;
                fitted_stages.insert(node, fitted);
                fitted_node_blocks.insert(node, output);
            }
            GraphNodePayload::Merge => {
                sk_validate_node_block(Some(node), &incoming_block, expected_rows)?;
                fitted_node_blocks.insert(node, incoming_block);
            }
            GraphNodePayload::FinalEstimator(estimator) => {
                let features_count = incoming_block.number_of_columns();
                let model = estimator
                    .fit_final_block(&incoming_block, targets)
                    .map_err(|error| SKPipelineGraphNodeError::node(node, error))?;
                fitted_node_blocks.insert(node, incoming_block);
                final_outcome = Some((model, features_count));
            }
        }
        Ok::<(), SKPipelineGraphNodeError>(())
    })?;

    let (final_model, number_of_features) = final_outcome.ok_or_else(|| {
        SKPipelineGraphNodeError::graph_level(SKError::InvalidHyperparameter {
            name: "final_estimator",
            reason: "pipeline graph execution did not encounter a final estimator node".to_string(),
        })
    })?;

    Ok(FittedGraphExecution {
        fitted_stages,
        incoming_widths,
        fitted_node_blocks,
        final_model,
        number_of_features,
    })
}

/// Transform `input_block` through every fitted non-final node in
/// `topological_order` and return the block entering `final_node`.
pub(super) fn sk_transform_graph<F: SKFloat>(
    graph: &DiGraph<GraphNodePayload<F>, usize>,
    topological_order: &[SKNodeIdentifier],
    fitted_stages: &BTreeMap<SKNodeIdentifier, SKBoxedFittedStage>,
    incoming_widths: &BTreeMap<SKNodeIdentifier, Vec<usize>>,
    final_node: SKNodeIdentifier,
    input_block: &SKFeatureBlock<F>,
) -> Result<SKFeatureBlock<F>, SKPipelineGraphNodeError> {
    let expected_rows = input_block.number_of_rows();
    sk_validate_node_block(None, input_block, expected_rows)?;

    let mut active_blocks = SKNodeBlockStore::new();
    let mut final_input: Option<SKFeatureBlock<F>> = None;
    topological_order.iter().try_for_each(|&node| {
        let expected_widths = incoming_widths.get(&node).map(Vec::as_slice);
        let (incoming_block, _) =
            sk_resolve_incoming_block(graph, node, input_block, &active_blocks, expected_widths)?;

        if node == final_node {
            final_input = Some(incoming_block);
            return Ok(());
        }

        let payload = &graph[NodeIndex::new(node.index())];
        let span = tracing::info_span!(
            "pipeline_graph_node",
            node_identifier = node.index(),
            node_kind = payload.kind().label(),
            operation = "transform"
        );
        let _guard = span.enter();

        match payload {
            GraphNodePayload::Transformer(stage) | GraphNodePayload::FeatureUnion(stage) => {
                let fitted = fitted_stages.get(&node).ok_or_else(|| {
                    SKPipelineGraphNodeError::node(
                        node,
                        SKError::Conversion("missing fitted state for graph node".to_string()),
                    )
                })?;
                let output = stage
                    .transform_block(fitted, &incoming_block)
                    .map_err(|error| SKPipelineGraphNodeError::node(node, error))?;
                sk_validate_node_block(Some(node), &output, expected_rows)?;
                active_blocks.insert(node, output);
            }
            GraphNodePayload::Merge => {
                sk_validate_node_block(Some(node), &incoming_block, expected_rows)?;
                active_blocks.insert(node, incoming_block);
            }
            GraphNodePayload::FinalEstimator(_) => {}
        }
        Ok::<(), SKPipelineGraphNodeError>(())
    })?;

    final_input.ok_or_else(|| {
        SKPipelineGraphNodeError::graph_level(SKError::Conversion(
            "pipeline graph transform did not reach the final estimator node".to_string(),
        ))
    })
}
