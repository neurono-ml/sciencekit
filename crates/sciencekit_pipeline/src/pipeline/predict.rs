//! Prediction: transform the incoming block through the fitted stages and
//! delegate to the final model.

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKClassifierPredictor, SKDataView, SKError, SKExecutionMode, SKFeatureBlock, SKFloat,
    SKOperationAttributes, SKOperationKind, SKRegressorPredictor, sk_run_operation,
};

use crate::stages::SKConfiguredStages;

use super::model::{SKPipelineClassifier, SKPipelineRegressor};

/// The shape of a data view, whichever representation it carries.
pub(super) fn sk_view_shape<F: SKFloat>(view: &SKDataView<'_, F>) -> (usize, usize) {
    match view {
        SKDataView::Dense(matrix) => (matrix.nrows(), matrix.ncols()),
        SKDataView::Sparse(matrix) => (matrix.rows(), matrix.cols()),
        _ => (0, 0),
    }
}

/// The span attributes shared by every pipeline prediction.
pub(super) fn sk_pipeline_attributes<F: SKFloat>(
    operation: SKOperationKind,
    view: &SKDataView<'_, F>,
    execution_intent: SKExecutionMode,
    backend: sciencekit_common::SKBackendKind,
) -> SKOperationAttributes {
    let (rows_count, columns_count) = sk_view_shape(view);
    SKOperationAttributes {
        operation,
        rows_count,
        columns_count,
        execution_mode: execution_intent,
        backend,
    }
}

/// Transform an incoming view through every fitted stage.
pub(super) fn sk_transform_input<F, Stages>(
    stages: &Stages,
    fitted_stages: &Stages::FittedStages,
    view: SKDataView<'_, F>,
) -> Result<SKFeatureBlock<F>, SKError>
where
    F: SKFloat,
    Stages: SKConfiguredStages<F>,
{
    let block = SKFeatureBlock::from(view);
    stages.transform_stages(fitted_stages, &block)
}

impl<F, Stages, FinalModel> SKClassifierPredictor<F> for SKPipelineClassifier<F, Stages, FinalModel>
where
    F: SKFloat,
    Stages: SKConfiguredStages<F>,
    FinalModel: SKClassifierPredictor<F>,
    FinalModel::Error: Into<SKError>,
{
    type Error = SKError;

    fn predict_labels<'a, X>(&self, features: X) -> Result<Array1<i64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view = features.try_into()?;
        let attributes = sk_pipeline_attributes(
            SKOperationKind::Predict,
            &view,
            self.execution_intent,
            self.backend,
        );
        sk_run_operation(attributes, || {
            let block = sk_transform_input(&self.stages, &self.fitted_stages, view)?;
            self.final_model.predict_labels(&block).map_err(Into::into)
        })
    }

    fn predict_probabilities<'a, X>(&self, features: X) -> Result<Array2<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view = features.try_into()?;
        let attributes = sk_pipeline_attributes(
            SKOperationKind::Predict,
            &view,
            self.execution_intent,
            self.backend,
        );
        sk_run_operation(attributes, || {
            let block = sk_transform_input(&self.stages, &self.fitted_stages, view)?;
            self.final_model
                .predict_probabilities(&block)
                .map_err(Into::into)
        })
    }
}

impl<F, Stages, FinalModel> SKRegressorPredictor<F> for SKPipelineRegressor<F, Stages, FinalModel>
where
    F: SKFloat,
    Stages: SKConfiguredStages<F>,
    FinalModel: SKRegressorPredictor<F>,
    FinalModel::Error: Into<SKError>,
{
    type Error = SKError;

    fn predict<'a, X>(&self, features: X) -> Result<Array1<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view = features.try_into()?;
        let attributes = sk_pipeline_attributes(
            SKOperationKind::Predict,
            &view,
            self.execution_intent,
            self.backend,
        );
        sk_run_operation(attributes, || {
            let block = sk_transform_input(&self.stages, &self.fitted_stages, view)?;
            self.final_model.predict(&block).map_err(Into::into)
        })
    }
}
