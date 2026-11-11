//! Mock dense-only and representation-spy base estimators for bagging
//! representation tests.

use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKClassifierPredictor, SKDataView, SKError, SKRegressorPredictor, SKSupervisedFit, SKTargetView,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum CustomDenseOnlyError {
    Central(String),
    RejectedRepresentation(&'static str),
}

impl fmt::Display for CustomDenseOnlyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Central(message) => formatter.write_str(message),
            Self::RejectedRepresentation(kind) => {
                write!(formatter, "dense-only base rejected `{kind}`")
            }
        }
    }
}

impl Error for CustomDenseOnlyError {}

impl From<SKError> for CustomDenseOnlyError {
    fn from(error: SKError) -> Self {
        Self::Central(error.to_string())
    }
}

#[derive(Clone, Debug)]
pub(super) struct CustomDenseOnlyClassifier;

#[derive(Clone, Debug)]
pub(super) struct CustomDenseOnlyModel;

impl SKSupervisedFit<f64> for CustomDenseOnlyClassifier {
    type Model = CustomDenseOnlyModel;
    type Error = CustomDenseOnlyError;

    fn fit<'a, X, T>(&self, features: X, _targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
        T: TryInto<SKTargetView<'a, f64>, Error = SKError>,
    {
        let view = features.try_into()?;
        match view {
            SKDataView::Dense(_) => Ok(CustomDenseOnlyModel),
            SKDataView::Sparse(_) => Err(CustomDenseOnlyError::RejectedRepresentation(
                view.representation(),
            )),
            _ => Err(CustomDenseOnlyError::RejectedRepresentation("unknown")),
        }
    }
}

impl SKClassifierPredictor<f64> for CustomDenseOnlyModel {
    type Error = CustomDenseOnlyError;

    fn predict_labels<'a, X>(&self, features: X) -> Result<Array1<i64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
    {
        let probabilities = self.predict_probabilities(features)?;
        Ok(Array1::zeros(probabilities.nrows()))
    }

    fn predict_probabilities<'a, X>(&self, features: X) -> Result<Array2<f64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
    {
        let view = features.try_into()?;
        match view {
            SKDataView::Dense(dense) => Ok(Array2::from_elem((dense.nrows(), 2), 0.5)),
            SKDataView::Sparse(_) => Err(CustomDenseOnlyError::RejectedRepresentation(
                view.representation(),
            )),
            _ => Err(CustomDenseOnlyError::RejectedRepresentation("unknown")),
        }
    }
}

#[derive(Clone, Debug)]
pub(super) struct RepresentationSpyRegressor {
    pub(super) seen_fit_representations: Arc<Mutex<Vec<&'static str>>>,
    pub(super) seen_predict_representations: Arc<Mutex<Vec<&'static str>>>,
}

#[derive(Clone, Debug)]
pub(super) struct RepresentationSpyModel {
    pub(super) seen_predict_representations: Arc<Mutex<Vec<&'static str>>>,
}

impl SKSupervisedFit<f64> for RepresentationSpyRegressor {
    type Model = RepresentationSpyModel;
    type Error = SKError;

    fn fit<'a, X, T>(&self, features: X, _targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
        T: TryInto<SKTargetView<'a, f64>, Error = SKError>,
    {
        let view = features.try_into()?;
        self.seen_fit_representations
            .lock()
            .unwrap()
            .push(view.representation());
        Ok(RepresentationSpyModel {
            seen_predict_representations: Arc::clone(&self.seen_predict_representations),
        })
    }
}

impl SKRegressorPredictor<f64> for RepresentationSpyModel {
    type Error = SKError;

    fn predict<'a, X>(&self, features: X) -> Result<Array1<f64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
    {
        let view = features.try_into()?;
        self.seen_predict_representations
            .lock()
            .unwrap()
            .push(view.representation());
        let rows = match view {
            SKDataView::Dense(dense) => dense.nrows(),
            SKDataView::Sparse(sparse) => sparse.rows(),
            _ => 0,
        };
        Ok(Array1::from_elem(rows, 1.0))
    }
}
