//! Regressor prediction wrapper [`SKPipelineGraphRegressor`] implementing
//! [`SKRegressorPredictor`] (spec `pipeline-graph`, Task 2.3).

use std::collections::BTreeMap;

use ndarray::Array1;
use sciencekit_common::{
    SKBackendKind, SKDataView, SKError, SKExecutionMode, SKFeatureBlock, SKFloat,
    SKRegressorPredictor, SKTargetView,
};

use super::errors::SKPipelineGraphNodeError;
use super::final_estimator::SKBoxedFinalModel;
use super::graph_definition::SKPipelineGraph;
use super::model::SKPipelineGraphModel;
use super::node::SKNodeIdentifier;

/// Fitted regressor DAG wrapper delegating [`SKRegressorPredictor`] through
/// an underlying [`SKPipelineGraphModel`].
pub struct SKPipelineGraphRegressor<F: SKFloat> {
    inner: SKPipelineGraphModel<F>,
}

impl<F: SKFloat> SKPipelineGraph<F> {
    /// Fit this graph and wrap the resulting model as an
    /// [`SKPipelineGraphRegressor`].
    pub fn fit_regressor<'a, X, T>(
        &self,
        features: X,
        targets: T,
    ) -> Result<SKPipelineGraphRegressor<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        self.fit_regressor_checked(features, targets)
            .map_err(Into::into)
    }

    /// Checked variant of [`Self::fit_regressor`] attributing node-level
    /// failures to their [`SKNodeIdentifier`].
    pub fn fit_regressor_checked<'a, X, T>(
        &self,
        features: X,
        targets: T,
    ) -> Result<SKPipelineGraphRegressor<F>, SKPipelineGraphNodeError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        let fitted = self.fit_checked(features, targets)?;
        let final_node = fitted.final_node();
        fitted
            .into_regressor()
            .map_err(|error| SKPipelineGraphNodeError::node(final_node, error))
    }
}

impl<F: SKFloat> SKPipelineGraphModel<F> {
    /// Convert into [`SKPipelineGraphRegressor`] if the final estimator was
    /// registered as a regressor.
    pub fn into_regressor(self) -> Result<SKPipelineGraphRegressor<F>, SKError> {
        if !self.final_model.is_regressor() {
            return Err(SKError::UnsupportedOperation {
                operation: "into_regressor",
                reason: "final estimator was not registered as a regressor".to_string(),
            });
        }
        Ok(SKPipelineGraphRegressor { inner: self })
    }
}

impl<F: SKFloat> SKPipelineGraphRegressor<F> {
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

    /// Checked response prediction attributing node-level failures to their
    /// [`SKNodeIdentifier`].
    pub fn predict_checked<'a, X>(&self, features: X) -> Result<Array1<F>, SKPipelineGraphNodeError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        self.inner.predict_checked(features)
    }
}

impl<F: SKFloat> SKRegressorPredictor<F> for SKPipelineGraphRegressor<F> {
    type Error = SKError;

    fn predict<'a, X>(&self, features: X) -> Result<Array1<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        self.inner.predict(features)
    }
}
