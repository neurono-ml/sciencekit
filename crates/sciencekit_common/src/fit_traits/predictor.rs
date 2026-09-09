//! Predictor contract.

use crate::SKError;
use crate::data_view::SKDataView;
use crate::sk_float::SKFloat;

/// Prediction lives on the fitted model type (never on the configured
/// estimator). Required by the convenient scoring forms, which run inference.
pub trait SKPredictor<F: SKFloat> {
    /// The inference error type; converts from the central [`SKError`].
    type Error: From<SKError>;
    /// Predict continuous scores/class indices for the given features.
    fn predict<'a, X>(&self, features: X) -> Result<ndarray::Array1<f64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>;
}
