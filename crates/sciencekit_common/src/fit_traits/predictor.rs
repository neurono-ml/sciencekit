//! Regressor and classifier prediction contracts.

use ndarray::{Array1, Array2};

use crate::SKError;
use crate::data_view::SKDataView;
use crate::sk_float::SKFloat;

/// Regressor prediction lives on the fitted model type (never on the
/// configured estimator). Responses preserve the model scalar end to end.
pub trait SKRegressorPredictor<F: SKFloat> {
    /// The inference error type; converts from the central [`SKError`].
    type Error: From<SKError>;
    /// Predict continuous responses for the given features.
    fn predict<'a, X>(&self, features: X) -> Result<Array1<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>;
}

/// Classifier prediction lives on the fitted model type (never on the
/// configured estimator). Labels are canonical `i64` indices independent of
/// the feature scalar; probabilities preserve the model scalar.
pub trait SKClassifierPredictor<F: SKFloat> {
    /// The inference error type; converts from the central [`SKError`].
    type Error: From<SKError>;
    /// Predict canonical label indices for the given features.
    fn predict_labels<'a, X>(&self, features: X) -> Result<Array1<i64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>;
    /// Predict class probabilities for the given features.
    fn predict_probabilities<'a, X>(&self, features: X) -> Result<Array2<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>;
}
