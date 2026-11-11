//! Classifier prediction wrapper [`SKPipelineGraphClassifier`] implementing
//! [`SKClassifierPredictor`] (spec `pipeline-graph`, Task 2.3).

use std::collections::BTreeMap;

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKBackendKind, SKClassifierPredictor, SKDataView, SKError, SKExecutionMode, SKFeatureBlock,
    SKFloat, SKTargetView,
};

use super::final_estimator::SKBoxedFinalModel;
use super::graph_definition::SKPipelineGraph;
use super::model::SKPipelineGraphModel;
use super::node::SKNodeIdentifier;

/// Fitted classifier DAG wrapper delegating [`SKClassifierPredictor`] through
/// an underlying [`SKPipelineGraphModel`].
pub struct SKPipelineGraphClassifier<F: SKFloat> {
    inner: SKPipelineGraphModel<F>,
}

impl<F: SKFloat> SKPipelineGraph<F> {
    /// Fit this graph and wrap the resulting model as an
    /// [`SKPipelineGraphClassifier`].
    pub fn fit_classifier<'a, X, T>(
        &self,
        features: X,
        targets: T,
    ) -> Result<SKPipelineGraphClassifier<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        self.fit(features, targets)?.into_classifier()
    }
}

impl<F: SKFloat> SKPipelineGraphModel<F> {
    /// Convert into [`SKPipelineGraphClassifier`] if the final estimator was
    /// registered as a classifier.
    pub fn into_classifier(self) -> Result<SKPipelineGraphClassifier<F>, SKError> {
        if !self.final_model.is_classifier() {
            return Err(SKError::UnsupportedOperation {
                operation: "into_classifier",
                reason: "final estimator was not registered as a classifier".to_string(),
            });
        }
        Ok(SKPipelineGraphClassifier { inner: self })
    }
}

impl<F: SKFloat> SKPipelineGraphClassifier<F> {
    /// Borrow the underlying [`SKPipelineGraphModel`].
    pub fn graph_model(&self) -> &SKPipelineGraphModel<F> {
        &self.inner
    }

    /// Unwrap into the underlying [`SKPipelineGraphModel`].
    pub fn into_graph_model(self) -> SKPipelineGraphModel<F> {
        self.inner
    }

    /// Number of nodes in the fitted graph.
    pub fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Number of directed edges in the fitted graph.
    pub fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    /// Deterministic topological execution order.
    pub fn topological_order(&self) -> &[SKNodeIdentifier] {
        self.inner.topological_order()
    }

    /// Fitted per-node feature blocks keyed by node identifier.
    pub fn fitted_node_blocks(&self) -> &BTreeMap<SKNodeIdentifier, SKFeatureBlock<F>> {
        self.inner.fitted_node_blocks()
    }

    /// Fitted final model handle.
    pub fn final_model(&self) -> &SKBoxedFinalModel<F> {
        self.inner.final_model()
    }

    /// Math backend recorded in operation spans.
    pub fn backend(&self) -> SKBackendKind {
        self.inner.backend()
    }

    /// Execution intent recorded in operation spans.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.inner.execution_mode()
    }
}

impl<F: SKFloat> SKClassifierPredictor<F> for SKPipelineGraphClassifier<F> {
    type Error = SKError;

    fn predict_labels<'a, X>(&self, features: X) -> Result<Array1<i64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        self.inner.predict_labels(features)
    }

    fn predict_probabilities<'a, X>(&self, features: X) -> Result<Array2<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        self.inner.predict_probabilities(features)
    }
}
