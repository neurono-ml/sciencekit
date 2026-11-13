//! The prediction facade over `SKNativeImport`: one surface, no type erasure.
//!
//! Every method delegates to the model's existing prediction trait, so the
//! imported model keeps its own numerics, observability span and dense/sparse
//! handling.

use ndarray::{Array1, Array2};
use sciencekit_common::{SKClusterLabel, SKDataView, SKError, SKFloat};

use super::delegation::{
    classifier_labels, classifier_probabilities, clusterer_predictions, regressor_predictions,
    transformer_transform,
};
use crate::native::conversion::SKNativeImport;
use crate::native::descriptor::SKNativeModelDescriptor;
use crate::native::model::SKNativeModelKind;

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
    pub fn kind(&self) -> SKNativeModelKind {
        match self {
            Self::Classifier { .. } => SKNativeModelKind::Classifier,
            Self::Regressor { .. } => SKNativeModelKind::Regressor,
            Self::Transformer { .. } => SKNativeModelKind::Transformer,
            Self::Clusterer { .. } => SKNativeModelKind::Clusterer,
        }
    }

    /// Predict continuous responses; only a regressor can answer.
    pub fn predict_regression<'a, X>(&self, features: X) -> Result<Array1<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let Self::Regressor { model, .. } = self else {
            return Err(wrong_surface("predict_regression", self.kind()));
        };
        regressor_predictions(model, features)
    }

    /// Predict canonical class indices; only a classifier can answer.
    pub fn predict_labels<'a, X>(&self, features: X) -> Result<Array1<i64>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let Self::Classifier { model, .. } = self else {
            return Err(wrong_surface("predict_labels", self.kind()));
        };
        classifier_labels(model, features)
    }

    /// Predict class probabilities; only a classifier can answer.
    pub fn predict_probabilities<'a, X>(&self, features: X) -> Result<Array2<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let Self::Classifier { model, .. } = self else {
            return Err(wrong_surface("predict_probabilities", self.kind()));
        };
        classifier_probabilities(model, features)
    }

    /// Transform features; only a transformer can answer.
    pub fn transform<'a, X>(&self, features: X) -> Result<Array2<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let Self::Transformer { model, .. } = self else {
            return Err(wrong_surface("transform", self.kind()));
        };
        transformer_transform(model, features)
    }

    /// Assign cluster labels; only a clusterer can answer.
    pub fn predict_clusters<'a, X>(&self, features: X) -> Result<Array1<SKClusterLabel>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let Self::Clusterer { model, .. } = self else {
            return Err(wrong_surface("predict_clusters", self.kind()));
        };
        clusterer_predictions(model, features)
    }
}

fn wrong_surface(operation: &'static str, kind: SKNativeModelKind) -> SKError {
    SKError::UnsupportedOperation {
        operation,
        reason: format!(
            "the imported model is a {}, not the requested surface",
            kind.name()
        ),
    }
}
