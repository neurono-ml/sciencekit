//! Inspection accessors and [`SKBuilder`] trait implementation for
//! [`SKPipelineGraphBuilder`] (spec `pipeline-graph`).

use petgraph::graph::NodeIndex;
use sciencekit_common::{SKBackendKind, SKBuilder, SKError, SKExecutionMode, SKFloat};

use super::builder::SKPipelineGraphBuilder;
use super::graph_definition::SKPipelineGraph;
use super::node::{GraphNodePayload, SKNodeIdentifier, SKPipelineGraphNodeKind};

impl<F: SKFloat> SKPipelineGraphBuilder<F> {
    /// Number of nodes currently added to the builder.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of valid directed edges currently added to the builder.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Kind of stage stored at `node`, if `node` belongs to this builder.
    pub fn node_kind(&self, node: SKNodeIdentifier) -> Option<SKPipelineGraphNodeKind> {
        self.graph
            .node_weight(NodeIndex::new(node.index()))
            .map(GraphNodePayload::kind)
    }

    /// Currently configured math backend.
    pub fn backend_kind(&self) -> SKBackendKind {
        self.backend
    }

    /// Currently configured execution intent.
    pub fn execution_intent(&self) -> SKExecutionMode {
        self.execution_intent
    }
}

impl<F: SKFloat> SKBuilder<SKPipelineGraph<F>> for SKPipelineGraphBuilder<F> {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        if let Ok(resolved) = mode.try_into() {
            self.execution_intent = resolved;
        }
        self
    }

    fn build(self) -> Result<SKPipelineGraph<F>, SKError> {
        SKPipelineGraphBuilder::build(self)
    }
}
