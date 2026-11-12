//! Representation-preserving prediction dispatch for the best model.

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKClassifierPredictor, SKDataView, SKError, SKFloat, SKRegressorPredictor,
};

/// The error returned when prediction is requested without a refitted model.
pub(crate) fn sk_missing_best_model(operation: &'static str) -> SKError {
    SKError::UnsupportedOperation {
        operation,
        reason: "the grid search was built with refit(false), so no best model is stored"
            .to_owned(),
    }
}

/// Predicts labels through the best model without changing representation.
pub(crate) fn sk_predict_labels<F, M>(
    model: &M,
    features: SKDataView<'_, F>,
) -> Result<Array1<i64>, SKError>
where
    F: SKFloat,
    M: SKClassifierPredictor<F>,
    SKError: From<M::Error>,
{
    match features {
        SKDataView::Dense(view) => model.predict_labels(view).map_err(SKError::from),
        SKDataView::Sparse(view) => model.predict_labels(view).map_err(SKError::from),
        _ => Err(sk_unsupported_prediction_representation()),
    }
}

/// Predicts class probabilities through the best model.
pub(crate) fn sk_predict_probabilities<F, M>(
    model: &M,
    features: SKDataView<'_, F>,
) -> Result<Array2<F>, SKError>
where
    F: SKFloat,
    M: SKClassifierPredictor<F>,
    SKError: From<M::Error>,
{
    match features {
        SKDataView::Dense(view) => model.predict_probabilities(view).map_err(SKError::from),
        SKDataView::Sparse(view) => model.predict_probabilities(view).map_err(SKError::from),
        _ => Err(sk_unsupported_prediction_representation()),
    }
}

/// Predicts continuous responses through the best model.
pub(crate) fn sk_predict_responses<F, M>(
    model: &M,
    features: SKDataView<'_, F>,
) -> Result<Array1<F>, SKError>
where
    F: SKFloat,
    M: SKRegressorPredictor<F>,
    SKError: From<M::Error>,
{
    match features {
        SKDataView::Dense(view) => model.predict(view).map_err(SKError::from),
        SKDataView::Sparse(view) => model.predict(view).map_err(SKError::from),
        _ => Err(sk_unsupported_prediction_representation()),
    }
}

fn sk_unsupported_prediction_representation() -> SKError {
    SKError::UnsupportedRepresentation {
        representation: "unknown",
        suggestion: "convert the features to dense or CSR",
    }
}
