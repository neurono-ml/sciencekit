//! Delegation from the facade to each concrete model's prediction trait.

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKClassifierPredictor, SKClusterLabel, SKClustererPredictor, SKDataView, SKError,
    SKFeatureTransformer, SKFloat, SKRegressorPredictor,
};

use crate::native::model::{
    SKNativeClassifier, SKNativeClusterer, SKNativeRegressor, SKNativeTransformer,
};

pub(super) fn regressor_predictions<'a, F: SKFloat, X>(
    model: &SKNativeRegressor<F>,
    features: X,
) -> Result<Array1<F>, SKError>
where
    X: TryInto<SKDataView<'a, F>, Error = SKError>,
{
    match model {
        SKNativeRegressor::LinearRegression(model) => {
            SKRegressorPredictor::predict(model, features)
        }
        SKNativeRegressor::Ridge(model) => SKRegressorPredictor::predict(model, features),
        SKNativeRegressor::StochasticGradient(model) => {
            SKRegressorPredictor::predict(model, features)
        }
        SKNativeRegressor::DecisionTree(model) => SKRegressorPredictor::predict(model, features),
        SKNativeRegressor::RandomForest(model) => SKRegressorPredictor::predict(model, features),
    }
}

pub(super) fn classifier_labels<'a, F: SKFloat, X>(
    model: &SKNativeClassifier<F>,
    features: X,
) -> Result<Array1<i64>, SKError>
where
    X: TryInto<SKDataView<'a, F>, Error = SKError>,
{
    match model {
        SKNativeClassifier::LogisticRegression(model) => {
            SKClassifierPredictor::predict_labels(model, features)
        }
        SKNativeClassifier::StochasticGradient(model) => {
            SKClassifierPredictor::predict_labels(model, features)
        }
        SKNativeClassifier::DecisionTree(model) => {
            SKClassifierPredictor::predict_labels(model, features)
        }
        SKNativeClassifier::RandomForest(model) => {
            SKClassifierPredictor::predict_labels(model, features)
        }
    }
}

pub(super) fn classifier_probabilities<'a, F: SKFloat, X>(
    model: &SKNativeClassifier<F>,
    features: X,
) -> Result<Array2<F>, SKError>
where
    X: TryInto<SKDataView<'a, F>, Error = SKError>,
{
    match model {
        SKNativeClassifier::LogisticRegression(model) => {
            SKClassifierPredictor::predict_probabilities(model, features)
        }
        SKNativeClassifier::StochasticGradient(model) => {
            SKClassifierPredictor::predict_probabilities(model, features)
        }
        SKNativeClassifier::DecisionTree(model) => {
            SKClassifierPredictor::predict_probabilities(model, features)
        }
        SKNativeClassifier::RandomForest(model) => {
            SKClassifierPredictor::predict_probabilities(model, features)
        }
    }
}

pub(super) fn transformer_transform<'a, F: SKFloat, X>(
    model: &SKNativeTransformer<F>,
    features: X,
) -> Result<Array2<F>, SKError>
where
    X: TryInto<SKDataView<'a, F>, Error = SKError>,
{
    match model {
        SKNativeTransformer::StandardScaler(model) => {
            SKFeatureTransformer::transform(model, features)
        }
        SKNativeTransformer::MinMaxScaler(model) => {
            SKFeatureTransformer::transform(model, features)
        }
        SKNativeTransformer::RobustScaler(model) => {
            SKFeatureTransformer::transform(model, features)
        }
    }
}

pub(super) fn clusterer_predictions<'a, F: SKFloat, X>(
    model: &SKNativeClusterer<F>,
    features: X,
) -> Result<Array1<SKClusterLabel>, SKError>
where
    X: TryInto<SKDataView<'a, F>, Error = SKError>,
{
    match model {
        SKNativeClusterer::KMeans(model) => SKClustererPredictor::predict_clusters(model, features),
    }
}
