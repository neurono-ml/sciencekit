//! Fixed-output models used for hand-computed scoring checks.

use ndarray::{Array1, Array2};
use sciencekit_common::{SKClassifierPredictor, SKDataView, SKError, SKRegressorPredictor};

/// A classifier model with fixed outputs.
#[derive(Debug, Clone)]
pub(crate) struct SKMockFixedClassifierModel {
    labels: Vec<i64>,
    probabilities: Vec<[f64; 2]>,
}

impl SKMockFixedClassifierModel {
    pub(crate) fn new(labels: Vec<i64>, probabilities: Vec<[f64; 2]>) -> Self {
        Self {
            labels,
            probabilities,
        }
    }
}

impl SKClassifierPredictor<f64> for SKMockFixedClassifierModel {
    type Error = SKError;

    fn predict_labels<'a, X>(&self, _features: X) -> Result<Array1<i64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
    {
        Ok(Array1::from(self.labels.clone()))
    }

    fn predict_probabilities<'a, X>(&self, _features: X) -> Result<Array2<f64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
    {
        Ok(Array2::from_shape_fn(
            (self.probabilities.len(), 2),
            |(row, column)| self.probabilities[row][column],
        ))
    }
}

/// A regressor model with fixed predictions.
#[derive(Debug, Clone)]
pub(crate) struct SKMockFixedRegressorModel {
    predictions: Vec<f64>,
}

impl SKMockFixedRegressorModel {
    pub(crate) fn new(predictions: Vec<f64>) -> Self {
        Self { predictions }
    }
}

impl SKRegressorPredictor<f64> for SKMockFixedRegressorModel {
    type Error = SKError;

    fn predict<'a, X>(&self, _features: X) -> Result<Array1<f64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
    {
        Ok(Array1::from(self.predictions.clone()))
    }
}
