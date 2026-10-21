//! Regressor prediction for the fitted lasso model (spec `lasso`).

use ndarray::Array1;
use sciencekit_common::{SKDataView, SKError, SKFloat, SKRegressorPredictor};

use super::SKLassoModel;

impl<F: SKFloat> SKRegressorPredictor<F> for SKLassoModel<F> {
    type Error = SKError;

    /// Predict the responses for the single configured `alpha`.
    ///
    /// With several alphas configured the coefficient vector is ambiguous, so
    /// the call returns a structured error directing the consumer to
    /// [`SKLassoModel::predict_with_alpha_index`].
    fn predict<'a, X>(&self, features: X) -> Result<Array1<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        if self.core.alphas().len() != 1 {
            return Err(self.core.ambiguous_prediction_error());
        }
        self.predict_with_alpha_index(0, features)
    }
}
