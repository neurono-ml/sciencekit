//! Validated [`SKPipelineGraph`] definition and graph structure accessors (spec
//! `pipeline-graph`).

use std::sync::Arc;

use petgraph::graph::{DiGraph, NodeIndex};
use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};

use super::builder::SKPipelineGraphBuilder;
use super::node::{GraphNodePayload, SKNodeIdentifier, SKPipelineGraphNodeKind};

/// Validated directed acyclic graph of feature stages ending in a single
/// supervised final estimator.
///
/// Constructed exclusively via [`SKPipelineGraph::build`] or
/// [`SKPipelineGraph::start`]; direct constructors are private.
pub struct SKPipelineGraph<F: SKFloat = f64> {
    pub(super) graph: Arc<DiGraph<GraphNodePayload<F>, usize>>,
    pub(super) topological_order: Vec<SKNodeIdentifier>,
    pub(super) final_node: SKNodeIdentifier,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
}

impl SKPipelineGraph<f64> {
    /// Start a DAG pipeline builder in scalar `F` with default `Faer` backend
    /// and `Automatic` execution mode.
    pub fn start<F: SKFloat>() -> SKPipelineGraphBuilder<F> {
        SKPipelineGraphBuilder::new()
    }
}

impl<F: SKFloat> SKPipelineGraph<F> {
    /// Start a DAG pipeline builder in scalar `F` with default `Faer` backend
    /// and `Automatic` execution mode.
    pub fn build() -> SKPipelineGraphBuilder<F> {
        SKPipelineGraphBuilder::new()
    }

    /// Number of nodes in the validated graph.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of directed edges in the validated graph.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Deterministic topological execution order of every node in the graph.
    pub fn topological_order(&self) -> &[SKNodeIdentifier] {
        &self.topological_order
    }

    /// Identifier of the unique final supervised estimator node.
    pub fn final_node(&self) -> SKNodeIdentifier {
        self.final_node
    }

    /// Kind of stage stored at `node`, if `node` belongs to this graph.
    pub fn node_kind(&self, node: SKNodeIdentifier) -> Option<SKPipelineGraphNodeKind> {
        self.graph
            .node_weight(NodeIndex::new(node.index()))
            .map(GraphNodePayload::kind)
    }

    /// Configured math backend recorded in every operation span.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// Configured execution intent recorded in every operation span.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.execution_intent
    }
}
