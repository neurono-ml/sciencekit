//! Regressor prediction for the fitted ridge model (spec `ridge-regression`).

use ndarray::Array1;
use sciencekit_common::{SKDataView, SKError, SKFloat, SKRegressorPredictor};

use super::SKRidgeModel;

impl<F: SKFloat> SKRegressorPredictor<F> for SKRidgeModel<F> {
    type Error = SKError;

    /// Predict the responses for the single configured `alpha`.
    ///
    /// With several alphas configured the coefficient vector is ambiguous, so
    /// the call returns a structured error directing the consumer to
    /// [`SKRidgeModel::predict_with_alpha_index`].
    fn predict<'a, X>(&self, features: X) -> Result<Array1<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        if self.alphas.len() != 1 {
            return Err(SKError::InvalidHyperparameter {
                name: "alpha",
                reason:
                    "several alphas were configured; use predict_with_alpha_index to select one"
                        .to_string(),
            });
        }
        self.predict_with_alpha_index(0, features)
    }
}
