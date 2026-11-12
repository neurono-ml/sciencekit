//! Predict, transform and score over a fitted model (spec `ffi-abi`).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sciencekit_common::{SKFeatureTransformer, SKFloat, SKRegressorPredictor};
use sciencekit_metrics::regression::sk_r2_score;

use crate::error_kind::SKFfiError;
use crate::model::definition::SKFfiFitted;

/// Predict responses with a fitted regressor.
pub fn sk_ffi_predict_fitted<F: SKFloat>(
    fitted: &SKFfiFitted<F>,
    features: ArrayView2<F>,
) -> Result<Array1<F>, SKFfiError> {
    match fitted {
        SKFfiFitted::LinearRegression(model) => Ok(model.predict(features)?),
        SKFfiFitted::StandardScaler(_) => Err(SKFfiError::Unsupported {
            operation: "predict",
            reason: "the standard scaler transforms features and does not predict".to_string(),
        }),
    }
}

/// Transform features with a fitted transformer.
pub fn sk_ffi_transform_fitted<F: SKFloat>(
    fitted: &SKFfiFitted<F>,
    features: ArrayView2<F>,
) -> Result<Array2<F>, SKFfiError> {
    match fitted {
        SKFfiFitted::StandardScaler(model) => Ok(model.transform(features)?),
        SKFfiFitted::LinearRegression(_) => Err(SKFfiError::Unsupported {
            operation: "transform",
            reason: "the linear-regression model predicts responses and does not transform"
                .to_string(),
        }),
    }
}

/// Score a fitted regressor with the coefficient of determination.
pub fn sk_ffi_score_fitted<F: SKFloat>(
    fitted: &SKFfiFitted<F>,
    features: ArrayView2<F>,
    targets: ArrayView1<F>,
) -> Result<F, SKFfiError> {
    let predictions = sk_ffi_predict_fitted(fitted, features)?;
    Ok(sk_r2_score(targets, predictions.view())?)
}
