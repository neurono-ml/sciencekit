//! Inference through fitted [`SKPipelineGraphModel`] (spec `pipeline-graph`,
//! Task 2.3).

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKDataView, SKError, SKFeatureBlock, SKFloat, SKOperationAttributes, SKOperationKind,
    sk_run_operation,
};

use super::errors::SKPipelineGraphNodeError;
use super::execute::sk_transform_graph;
use super::model::{SKPipelineGraphModel, view_dimensions};

impl<F: SKFloat> SKPipelineGraphModel<F> {
    /// Transform `features` through every fitted non-final node in topological
    /// order and return the feature block entering `final_node`.
    pub fn transform_to_final<'a, X>(&self, features: X) -> Result<SKFeatureBlock<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        self.transform_checked(features).map_err(Into::into)
    }

    /// Checked variant of [`Self::transform_to_final`] attributing node-level
    /// failures to their [`super::SKNodeIdentifier`].
    pub fn transform_checked<'a, X>(
        &self,
        features: X,
    ) -> Result<SKFeatureBlock<F>, SKPipelineGraphNodeError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view = features.try_into()?;
        let (rows_count, columns_count) = view_dimensions(&view);
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Transform,
            rows_count,
            columns_count,
            execution_mode: self.execution_intent,
            backend: self.backend,
        };
        sk_run_operation(attributes, || self.execute_transform_view(view))
    }

    /// Predict canonical label indices through the fitted graph and classifier
    /// final model.
    pub fn predict_labels<'a, X>(&self, features: X) -> Result<Array1<i64>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        self.predict_labels_checked(features).map_err(Into::into)
    }

    /// Checked variant of [`Self::predict_labels`].
    pub fn predict_labels_checked<'a, X>(
        &self,
        features: X,
    ) -> Result<Array1<i64>, SKPipelineGraphNodeError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view = features.try_into()?;
        let attributes = self.predict_attributes(&view);
        sk_run_operation(attributes, || {
            let final_block = self.execute_transform_view(view)?;
            self.final_model
                .predict_labels(&final_block)
                .map_err(|error| SKPipelineGraphNodeError::node(self.final_node, error))
        })
    }

    /// Predict class probabilities through the fitted graph and classifier
    /// final model.
    pub fn predict_probabilities<'a, X>(&self, features: X) -> Result<Array2<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        self.predict_probabilities_checked(features)
            .map_err(Into::into)
    }

    /// Checked variant of [`Self::predict_probabilities`].
    pub fn predict_probabilities_checked<'a, X>(
        &self,
        features: X,
    ) -> Result<Array2<F>, SKPipelineGraphNodeError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view = features.try_into()?;
        let attributes = self.predict_attributes(&view);
        sk_run_operation(attributes, || {
            let final_block = self.execute_transform_view(view)?;
            self.final_model
                .predict_probabilities(&final_block)
                .map_err(|error| SKPipelineGraphNodeError::node(self.final_node, error))
        })
    }

    /// Predict continuous responses through the fitted graph and regressor
    /// final model.
    pub fn predict<'a, X>(&self, features: X) -> Result<Array1<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        self.predict_checked(features).map_err(Into::into)
    }

    /// Checked variant of [`Self::predict`].
    pub fn predict_checked<'a, X>(&self, features: X) -> Result<Array1<F>, SKPipelineGraphNodeError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view = features.try_into()?;
        let attributes = self.predict_attributes(&view);
        sk_run_operation(attributes, || {
            let final_block = self.execute_transform_view(view)?;
            self.final_model
                .predict(&final_block)
                .map_err(|error| SKPipelineGraphNodeError::node(self.final_node, error))
        })
    }

    fn predict_attributes(&self, view: &SKDataView<'_, F>) -> SKOperationAttributes {
        let (rows_count, columns_count) = view_dimensions(view);
        SKOperationAttributes {
            operation: SKOperationKind::Predict,
            rows_count,
            columns_count,
            execution_mode: self.execution_intent,
            backend: self.backend,
        }
    }

    fn execute_transform_view(
        &self,
        view: SKDataView<'_, F>,
    ) -> Result<SKFeatureBlock<F>, SKPipelineGraphNodeError> {
        let (rows_count, columns_count) = view_dimensions(&view);
        if columns_count != self.number_of_input_features {
            return Err(SKPipelineGraphNodeError::graph_level(
                SKError::ShapeMismatch {
                    expected: vec![rows_count, self.number_of_input_features],
                    found: vec![rows_count, columns_count],
                },
            ));
        }
        let input_block = SKFeatureBlock::from(view);
        sk_transform_graph(
            &self.graph,
            &self.topological_order,
            &self.fitted_stages,
            &self.incoming_widths,
            self.final_node,
            &input_block,
        )
    }
}
