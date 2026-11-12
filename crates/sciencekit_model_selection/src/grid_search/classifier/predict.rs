//! Classifier prediction delegating to the best fitted model
//! (spec `grid-search`, requirement `Classifier and regressor prediction`).

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKClassifierPredictor, SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind,
    SKSupervisedFit, sk_run_operation,
};

use super::model::SKGridSearchCVClassifierModel;
use crate::grid_search::observation::sk_view_dimensions;
use crate::grid_search::prediction::{
    sk_missing_best_model, sk_predict_labels, sk_predict_probabilities,
};

impl<F, E> SKClassifierPredictor<F> for SKGridSearchCVClassifierModel<F, E>
where
    F: SKFloat,
    E: SKSupervisedFit<F>,
    E::Model: SKClassifierPredictor<F>,
    SKError: From<<E::Model as SKClassifierPredictor<F>>::Error>,
{
    type Error = SKError;

    fn predict_labels<'a, X>(&self, features: X) -> Result<Array1<i64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        self.sk_with_best_model("predict_labels", features, sk_predict_labels)
    }

    fn predict_probabilities<'a, X>(&self, features: X) -> Result<Array2<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        self.sk_with_best_model("predict_probabilities", features, sk_predict_probabilities)
    }
}

impl<F, E> SKGridSearchCVClassifierModel<F, E>
where
    F: SKFloat,
    E: SKSupervisedFit<F>,
    E::Model: SKClassifierPredictor<F>,
    SKError: From<<E::Model as SKClassifierPredictor<F>>::Error>,
{
    fn sk_with_best_model<'a, X, R>(
        &self,
        operation: &'static str,
        features: X,
        predict: impl FnOnce(&E::Model, SKDataView<'_, F>) -> Result<R, SKError>,
    ) -> Result<R, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        let (rows_count, columns_count) = sk_view_dimensions(features);
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Predict,
            rows_count,
            columns_count,
            execution_mode: self.execution_mode,
            backend: self.backend,
        };
        sk_run_operation(attributes, || {
            let model = self
                .best_model
                .as_ref()
                .ok_or_else(|| sk_missing_best_model(operation))?;
            predict(model, features)
        })
    }
}
