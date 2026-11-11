//! Fixtures for bagging validation tests.

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKDataView, SKError, SKRegressorPredictor, SKSupervisedFit,
    SKTargetView,
};
use sciencekit_tree::{SKDecisionTreeClassifier, SKDecisionTreeRegressor};

use crate::{SKBaggingClassifier, SKBaggingRegressor};

pub(super) fn default_classifier() -> SKBaggingClassifier<SKDecisionTreeClassifier<f64>> {
    SKBaggingClassifier::build(SKDecisionTreeClassifier::<f64>::build().build().unwrap())
        .build()
        .unwrap()
}

pub(super) fn default_regressor() -> SKBaggingRegressor<SKDecisionTreeRegressor<f64>> {
    SKBaggingRegressor::build(SKDecisionTreeRegressor::<f64>::build().build().unwrap())
        .build()
        .unwrap()
}

#[derive(Clone, Debug)]
pub(super) struct InvalidOutputBase(pub(super) f64);

impl SKSupervisedFit<f64> for InvalidOutputBase {
    type Model = Self;
    type Error = SKError;

    fn fit<'a, X, T>(&self, _features: X, _targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
        T: TryInto<SKTargetView<'a, f64>, Error = SKError>,
    {
        Ok(self.clone())
    }
}

impl SKClassifierPredictor<f64> for InvalidOutputBase {
    type Error = SKError;

    fn predict_labels<'a, X>(&self, features: X) -> Result<Array1<i64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
    {
        Ok(Array1::zeros(self.predict_probabilities(features)?.nrows()))
    }

    fn predict_probabilities<'a, X>(&self, features: X) -> Result<Array2<f64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
    {
        let rows = match features.try_into()? {
            SKDataView::Dense(dense) => dense.nrows(),
            SKDataView::Sparse(sparse) => sparse.rows(),
            _ => 0,
        };
        Ok(Array2::from_shape_fn((rows, 2), |(_, column)| {
            if column == 0 { self.0 } else { 1.0 - self.0 }
        }))
    }
}

impl SKRegressorPredictor<f64> for InvalidOutputBase {
    type Error = SKError;

    fn predict<'a, X>(&self, features: X) -> Result<Array1<f64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
    {
        let rows = match features.try_into()? {
            SKDataView::Dense(dense) => dense.nrows(),
            SKDataView::Sparse(sparse) => sparse.rows(),
            _ => 0,
        };
        Ok(Array1::from_elem(rows, self.0))
    }
}
