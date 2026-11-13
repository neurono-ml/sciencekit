//! Uniform prediction over the reconstructed native models
//! (spec `native-model-import`, requirement "Safetensors native import").
//!
//! One facade hides the concrete model type without erasing it: every method
//! delegates to the model's existing prediction trait, so the imported model
//! keeps its own numerics, observability span and dense/sparse handling.

use ndarray::{Array1, Array2, ArrayView2};
use sciencekit_common::{
    SKClassifierPredictor, SKClusterLabel, SKClustererPredictor, SKError, SKFeatureTransformer,
    SKFloat, SKRegressorPredictor,
};

use super::conversion::SKNativeImport;
use super::descriptor::SKNativeModelDescriptor;
use super::model::{SKNativeClassifier, SKNativeClusterer, SKNativeRegressor, SKNativeTransformer};

impl<F: SKFloat> SKNativeImport<F> {
    /// The metadata the conversion preserved.
    pub fn descriptor(&self) -> &SKNativeModelDescriptor {
        match self {
            Self::Classifier { descriptor, .. }
            | Self::Regressor { descriptor, .. }
            | Self::Transformer { descriptor, .. }
            | Self::Clusterer { descriptor, .. } => descriptor,
        }
    }

    /// Whether this model can continue training on more data.
    pub fn supports_retraining(&self) -> bool {
        self.descriptor().supports_retraining()
    }

    /// The prediction surface this model exposes.
    pub fn kind(&self) -> super::model::SKNativeModelKind {
        match self {
            Self::Classifier { .. } => super::model::SKNativeModelKind::Classifier,
            Self::Regressor { .. } => super::model::SKNativeModelKind::Regressor,
            Self::Transformer { .. } => super::model::SKNativeModelKind::Transformer,
            Self::Clusterer { .. } => super::model::SKNativeModelKind::Clusterer,
        }
    }

    /// Predict continuous responses; only a regressor can answer.
    pub fn predict_regression(&self, features: ArrayView2<'_, F>) -> Result<Array1<F>, SKError> {
        let Self::Regressor { model, .. } = self else {
            return Err(wrong_surface("predict_regression", self.kind()));
        };
        regressor_predict(model, features)
    }

    /// Predict canonical class indices; only a classifier can answer.
    pub fn predict_labels(&self, features: ArrayView2<'_, F>) -> Result<Array1<i64>, SKError> {
        let Self::Classifier { model, .. } = self else {
            return Err(wrong_surface("predict_labels", self.kind()));
        };
        classifier_labels(model, features)
    }

    /// Predict class probabilities; only a classifier can answer.
    pub fn predict_probabilities(&self, features: ArrayView2<'_, F>) -> Result<Array2<F>, SKError> {
        let Self::Classifier { model, .. } = self else {
            return Err(wrong_surface("predict_probabilities", self.kind()));
        };
        classifier_probabilities(model, features)
    }

    /// Transform features; only a transformer can answer.
    pub fn transform(&self, features: ArrayView2<'_, F>) -> Result<Array2<F>, SKError> {
        let Self::Transformer { model, .. } = self else {
            return Err(wrong_surface("transform", self.kind()));
        };
        transformer_transform(model, features)
    }

    /// Assign cluster labels; only a clusterer can answer.
    pub fn predict_clusters(
        &self,
        features: ArrayView2<'_, F>,
    ) -> Result<Array1<SKClusterLabel>, SKError> {
        let Self::Clusterer { model, .. } = self else {
            return Err(wrong_surface("predict_clusters", self.kind()));
        };
        clusterer_clusters(model, features)
    }
}

fn wrong_surface(operation: &'static str, kind: super::model::SKNativeModelKind) -> SKError {
    SKError::UnsupportedOperation {
        operation,
        reason: format!(
            "the imported model is a {}, not the requested surface",
            kind.name()
        ),
    }
}

fn regressor_predict<F: SKFloat>(
    model: &SKNativeRegressor<F>,
    features: ArrayView2<'_, F>,
) -> Result<Array1<F>, SKError> {
    match model {
        SKNativeRegressor::LinearRegression(model) => {
            SKRegressorPredictor::predict(model, features).map_err(Into::into)
        }
        SKNativeRegressor::Ridge(model) => {
            SKRegressorPredictor::predict(model, features).map_err(Into::into)
        }
        SKNativeRegressor::StochasticGradient(model) => {
            SKRegressorPredictor::predict(model, features).map_err(Into::into)
        }
        SKNativeRegressor::DecisionTree(model) => {
            SKRegressorPredictor::predict(model, features).map_err(Into::into)
        }
        SKNativeRegressor::RandomForest(model) => {
            SKRegressorPredictor::predict(model, features).map_err(Into::into)
        }
    }
}

fn classifier_labels<F: SKFloat>(
    model: &SKNativeClassifier<F>,
    features: ArrayView2<'_, F>,
) -> Result<Array1<i64>, SKError> {
    match model {
        SKNativeClassifier::LogisticRegression(model) => {
            SKClassifierPredictor::predict_labels(model, features).map_err(Into::into)
        }
        SKNativeClassifier::StochasticGradient(model) => {
            SKClassifierPredictor::predict_labels(model, features).map_err(Into::into)
        }
        SKNativeClassifier::DecisionTree(model) => {
            SKClassifierPredictor::predict_labels(model, features).map_err(Into::into)
        }
        SKNativeClassifier::RandomForest(model) => {
            SKClassifierPredictor::predict_labels(model, features).map_err(Into::into)
        }
    }
}

fn classifier_probabilities<F: SKFloat>(
    model: &SKNativeClassifier<F>,
    features: ArrayView2<'_, F>,
) -> Result<Array2<F>, SKError> {
    match model {
        SKNativeClassifier::LogisticRegression(model) => {
            SKClassifierPredictor::predict_probabilities(model, features).map_err(Into::into)
        }
        SKNativeClassifier::StochasticGradient(model) => {
            SKClassifierPredictor::predict_probabilities(model, features).map_err(Into::into)
        }
        SKNativeClassifier::DecisionTree(model) => {
            SKClassifierPredictor::predict_probabilities(model, features).map_err(Into::into)
        }
        SKNativeClassifier::RandomForest(model) => {
            SKClassifierPredictor::predict_probabilities(model, features).map_err(Into::into)
        }
    }
}

fn transformer_transform<F: SKFloat>(
    model: &SKNativeTransformer<F>,
    features: ArrayView2<'_, F>,
) -> Result<Array2<F>, SKError> {
    match model {
        SKNativeTransformer::StandardScaler(model) => {
            SKFeatureTransformer::transform(model, features).map_err(Into::into)
        }
        SKNativeTransformer::MinMaxScaler(model) => {
            SKFeatureTransformer::transform(model, features).map_err(Into::into)
        }
        SKNativeTransformer::RobustScaler(model) => {
            SKFeatureTransformer::transform(model, features).map_err(Into::into)
        }
    }
}

fn clusterer_clusters<F: SKFloat>(
    model: &SKNativeClusterer<F>,
    features: ArrayView2<'_, F>,
) -> Result<Array1<SKClusterLabel>, SKError> {
    match model {
        SKNativeClusterer::KMeans(model) => {
            SKClustererPredictor::predict_clusters(model, features).map_err(Into::into)
        }
    }
}
