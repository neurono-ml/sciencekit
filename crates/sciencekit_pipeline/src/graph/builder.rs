//! Mandatory builder [`SKPipelineGraphBuilder`] for [`SKPipelineGraph`] (spec
//! `pipeline-graph`).

use std::sync::Arc;

use petgraph::graph::{DiGraph, NodeIndex};
use sciencekit_common::{
    SKBackendKind, SKClassifierPredictor, SKError, SKExecutionMode, SKFloat, SKRegressorPredictor,
    SKSupervisedFit,
};

use crate::{SKBlockTransformer, SKFeatureUnion};

use super::final_estimator::{
    ClassifierFinalAdapter, RegressorFinalAdapter, SupervisedFinalAdapter,
};
use super::graph_definition::SKPipelineGraph;
use super::node::{GraphNodePayload, SKNodeIdentifier};
use super::validate::sk_validate_graph;

/// Mandatory builder for [`SKPipelineGraph`].
pub struct SKPipelineGraphBuilder<F: SKFloat> {
    pub(super) graph: DiGraph<GraphNodePayload<F>, usize>,
    pub(super) next_edge_insertion_order: usize,
    pub(super) invalid_endpoints: Vec<(SKNodeIdentifier, SKNodeIdentifier)>,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
}

impl<F: SKFloat> SKPipelineGraphBuilder<F> {
    pub(super) fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            next_edge_insertion_order: 0,
            invalid_endpoints: Vec::new(),
            backend: SKBackendKind::Faer,
            execution_intent: SKExecutionMode::Automatic,
        }
    }

    /// Add an unsupervised block transformer node and return its identifier.
    ///
    /// Every node shares the scalar `F`; inserting a stage implemented only for
    /// a different scalar fails at compile time:
    ///
    /// ```compile_fail
    /// use sciencekit_common::SKError;
    /// use sciencekit_pipeline::{SKFeatureBlock, SKPipelineGraph, SKPipelineTransformer};
    ///
    /// struct F64OnlyStage;
    /// impl SKPipelineTransformer<f64> for F64OnlyStage {
    ///     type Fitted = ();
    ///     fn fit_stage(&self, _block: &SKFeatureBlock<f64>) -> Result<(), SKError> { Ok(()) }
    ///     fn transform_stage(&self, _fitted: &(), block: &SKFeatureBlock<f64>) -> Result<SKFeatureBlock<f64>, SKError> {
    ///         Ok(block.clone())
    ///     }
    /// }
    ///
    /// let mut builder = SKPipelineGraph::start::<f32>();
    /// let _ = builder.add_transformer(F64OnlyStage);
    /// ```
    pub fn add_transformer<Stage>(&mut self, stage: Stage) -> SKNodeIdentifier
    where
        Stage: SKBlockTransformer<F> + 'static,
    {
        self.push_node(GraphNodePayload::Transformer(Box::new(stage)))
    }

    /// Add a [`SKFeatureUnion`] node and return its identifier.
    pub fn add_feature_union<Stages>(
        &mut self,
        feature_union: SKFeatureUnion<Stages>,
    ) -> SKNodeIdentifier
    where
        SKFeatureUnion<Stages>: SKBlockTransformer<F> + 'static,
    {
        self.push_node(GraphNodePayload::FeatureUnion(Box::new(feature_union)))
    }

    /// Add an explicit horizontal-concatenation merge node.
    pub fn add_merge(&mut self) -> SKNodeIdentifier {
        self.push_node(GraphNodePayload::Merge)
    }

    /// Add a supervised classifier as the final estimator node.
    pub fn add_classifier<Estimator>(&mut self, estimator: Estimator) -> SKNodeIdentifier
    where
        Estimator: SKSupervisedFit<F> + Send + Sync + 'static,
        Estimator::Error: Into<SKError>,
        Estimator::Model: SKClassifierPredictor<F> + Send + Sync + 'static,
        <Estimator::Model as SKClassifierPredictor<F>>::Error: Into<SKError>,
    {
        self.push_node(GraphNodePayload::FinalEstimator(Box::new(
            ClassifierFinalAdapter(estimator),
        )))
    }

    /// Add a supervised regressor as the final estimator node.
    pub fn add_regressor<Estimator>(&mut self, estimator: Estimator) -> SKNodeIdentifier
    where
        Estimator: SKSupervisedFit<F> + Send + Sync + 'static,
        Estimator::Error: Into<SKError>,
        Estimator::Model: SKRegressorPredictor<F> + Send + Sync + 'static,
        <Estimator::Model as SKRegressorPredictor<F>>::Error: Into<SKError>,
    {
        self.push_node(GraphNodePayload::FinalEstimator(Box::new(
            RegressorFinalAdapter(estimator),
        )))
    }

    /// Add a general supervised estimator as the final estimator node.
    pub fn add_final_estimator<Estimator>(&mut self, estimator: Estimator) -> SKNodeIdentifier
    where
        Estimator: SKSupervisedFit<F> + Send + Sync + 'static,
        Estimator::Error: Into<SKError>,
        Estimator::Model: Send + Sync + 'static,
    {
        self.push_node(GraphNodePayload::FinalEstimator(Box::new(
            SupervisedFinalAdapter(estimator),
        )))
    }

    /// Insert a directed feature-block flow edge from `source` to `target`.
    pub fn connect(&mut self, source: SKNodeIdentifier, target: SKNodeIdentifier) -> &mut Self {
        let count = self.graph.node_count();
        if source.index() >= count || target.index() >= count {
            self.invalid_endpoints.push((source, target));
            return self;
        }
        let order = self.next_edge_insertion_order;
        self.next_edge_insertion_order += 1;
        self.graph.add_edge(
            NodeIndex::new(source.index()),
            NodeIndex::new(target.index()),
            order,
        );
        self
    }

    /// Set the backend kind recorded in every operation span (default `Faer`).
    pub fn backend(mut self, backend: SKBackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Set the execution intent recorded in every operation span (default
    /// `Automatic`).
    pub fn execution_mode<E>(mut self, mode: E) -> Self
    where
        E: TryInto<SKExecutionMode>,
    {
        if let Ok(resolved) = mode.try_into() {
            self.execution_intent = resolved;
        }
        self
    }

    /// Validate acyclicity, single final estimator sink and full reachability.
    pub fn validate(&self) -> Result<(), SKError> {
        sk_validate_graph(&self.graph, &self.invalid_endpoints).map(|_| ())
    }

    /// Validate and construct a [`SKPipelineGraph`] from this builder.
    pub fn build(self) -> Result<SKPipelineGraph<F>, SKError> {
        let final_node = sk_validate_graph(&self.graph, &self.invalid_endpoints)?;
        let topological_order = (0..self.graph.node_count())
            .map(SKNodeIdentifier::new)
            .collect();
        Ok(SKPipelineGraph {
            graph: Arc::new(self.graph),
            topological_order,
            final_node,
            backend: self.backend,
            execution_intent: self.execution_intent,
        })
    }

    fn push_node(&mut self, payload: GraphNodePayload<F>) -> SKNodeIdentifier {
        SKNodeIdentifier::new(self.graph.add_node(payload).index())
    }
}
