//! Prediction delegation for [`SKHalvingGridSearchRegressorModel`].

use ndarray::Array1;
use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKRegressorPredictor,
    SKSupervisedFit, sk_run_operation,
};

use crate::grid_search::observation::sk_view_dimensions;
use crate::grid_search::prediction::sk_predict_responses;

use super::model::SKHalvingGridSearchRegressorModel;

impl<F, E> SKRegressorPredictor<F> for SKHalvingGridSearchRegressorModel<F, E>
where
    F: SKFloat,
    E: SKSupervisedFit<F>,
    E::Model: SKRegressorPredictor<F>,
    SKError: From<<E::Model as SKRegressorPredictor<F>>::Error>,
{
    type Error = SKError;

    fn predict<'a, X>(&self, features: X) -> Result<Array1<F>, Self::Error>
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
                .ok_or_else(|| SKError::UnsupportedOperation {
                    operation: "predict",
                    reason:
                        "the halving grid search was built with refit(false), so no best model is stored"
                            .to_owned(),
                })?;
            sk_predict_responses(model, features)
        })
    }
}
