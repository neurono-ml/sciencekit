//! Fitted [`SKPipelineGraphModel`] and [`SKPipelineGraph::fit`] execution (spec
//! `pipeline-graph`).

use std::collections::BTreeMap;
use std::sync::Arc;

use petgraph::graph::DiGraph;
use sciencekit_common::{
    SKBackendKind, SKDataView, SKError, SKExecutionMode, SKFeatureBlock, SKFloat,
    SKOperationAttributes, SKOperationKind, SKSupervisedFit, SKTargetView, sk_run_operation,
};

use crate::SKBoxedFittedStage;

use super::errors::SKPipelineGraphNodeError;
use super::execute::sk_fit_graph;
use super::final_estimator::SKBoxedFinalModel;
use super::graph_definition::SKPipelineGraph;
use super::node::{GraphNodePayload, SKNodeIdentifier};
use super::order::SKNodeBlockStore;

/// Immutable fitted DAG pipeline storing fitted transformer stages, per-node
/// fitted feature blocks and the fitted final supervised model.
pub struct SKPipelineGraphModel<F: SKFloat> {
    pub(super) graph: Arc<DiGraph<GraphNodePayload<F>, usize>>,
    pub(super) topological_order: Vec<SKNodeIdentifier>,
    pub(super) final_node: SKNodeIdentifier,
    #[allow(dead_code)]
    pub(super) fitted_stages: BTreeMap<SKNodeIdentifier, SKBoxedFittedStage>,
    #[allow(dead_code)]
    pub(super) incoming_widths: BTreeMap<SKNodeIdentifier, Vec<usize>>,
    pub(super) fitted_node_blocks: SKNodeBlockStore<F>,
    pub(super) final_model: SKBoxedFinalModel<F>,
    pub(super) number_of_input_features: usize,
    pub(super) number_of_features: usize,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
}

impl<F: SKFloat> SKPipelineGraph<F> {
    /// Fit every node in topological order on `features` and `targets`.
    pub fn fit<'a, X, T>(&self, features: X, targets: T) -> Result<SKPipelineGraphModel<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        self.fit_checked(features, targets).map_err(Into::into)
    }

    /// Fit every node in topological order, pairing any node-level failure with
    /// its [`SKNodeIdentifier`].
    pub fn fit_checked<'a, X, T>(
        &self,
        features: X,
        targets: T,
    ) -> Result<SKPipelineGraphModel<F>, SKPipelineGraphNodeError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        let view = features.try_into()?;
        let targets = targets.try_into()?;
        let (rows_count, columns_count) = view_dimensions(&view);
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Fit,
            rows_count,
            columns_count,
            execution_mode: self.execution_intent,
            backend: self.backend,
        };
        sk_run_operation(attributes, || {
            let input_block = SKFeatureBlock::from(view);
            let execution =
                sk_fit_graph(&self.graph, &self.topological_order, &input_block, targets)?;
            Ok(SKPipelineGraphModel {
                graph: Arc::clone(&self.graph),
                topological_order: self.topological_order.clone(),
                final_node: self.final_node,
                fitted_stages: execution.fitted_stages,
                incoming_widths: execution.incoming_widths,
                fitted_node_blocks: execution.fitted_node_blocks,
                final_model: execution.final_model,
                number_of_input_features: columns_count,
                number_of_features: execution.number_of_features,
                backend: self.backend,
                execution_intent: self.execution_intent,
            })
        })
    }
}

impl<F: SKFloat> SKSupervisedFit<F> for SKPipelineGraph<F> {
    type Model = SKPipelineGraphModel<F>;
    type Error = SKError;

    fn fit<'a, X, T>(&self, features: X, targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        SKPipelineGraph::fit(self, features, targets)
    }
}

impl<F: SKFloat> SKPipelineGraphModel<F> {
    /// Number of nodes in the fitted graph.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Number of directed edges in the fitted graph.
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

    /// Ordered map of fitted per-node feature blocks keyed by node identifier.
    pub fn fitted_node_blocks(&self) -> &BTreeMap<SKNodeIdentifier, SKFeatureBlock<F>> {
        self.fitted_node_blocks.as_map()
    }

    /// Borrow the fitted feature block recorded for `node`, if present.
    pub fn fitted_node_block(&self, node: SKNodeIdentifier) -> Option<&SKFeatureBlock<F>> {
        self.fitted_node_blocks.get(node)
    }

    /// Borrow the type-erased fitted final supervised model.
    pub fn final_model(&self) -> &SKBoxedFinalModel<F> {
        &self.final_model
    }

    /// Number of input features delivered to root nodes during `fit`.
    pub fn number_of_input_features(&self) -> usize {
        self.number_of_input_features
    }

    /// Number of features the final estimator was fitted on.
    pub fn number_of_features(&self) -> usize {
        self.number_of_features
    }

    /// Math backend inherited from [`SKPipelineGraph`].
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// Execution intent inherited from [`SKPipelineGraph`].
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.execution_intent
    }
}

pub(super) fn view_dimensions<F: SKFloat>(view: &SKDataView<'_, F>) -> (usize, usize) {
    match view {
        SKDataView::Dense(matrix) => (matrix.nrows(), matrix.ncols()),
        SKDataView::Sparse(matrix) => (matrix.rows(), matrix.cols()),
        _ => (0, 0),
    }
}
