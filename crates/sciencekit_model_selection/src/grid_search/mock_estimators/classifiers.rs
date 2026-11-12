//! Threshold mock classifier.

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKClassifierPredictor, SKDataView, SKError, SKSupervisedFit, SKTargetView,
};

use super::helpers::{sk_first_column, sk_number_of_rows};

/// A classifier deciding label `1` when the first feature reaches a threshold.
#[derive(Debug, Clone)]
pub(crate) struct SKMockThresholdClassifier {
    threshold: f64,
}

impl SKMockThresholdClassifier {
    pub(crate) fn new(threshold: f64) -> Self {
        Self { threshold }
    }

    pub(crate) fn threshold(&self) -> f64 {
        self.threshold
    }
}

/// The fitted threshold classifier.
#[derive(Debug, Clone)]
pub(crate) struct SKMockThresholdClassifierModel {
    threshold: f64,
    training_rows: usize,
}

impl SKMockThresholdClassifierModel {
    pub(crate) fn training_rows(&self) -> usize {
        self.training_rows
    }
}

impl SKSupervisedFit<f64> for SKMockThresholdClassifier {
    type Model = SKMockThresholdClassifierModel;
    type Error = SKError;

    fn fit<'a, X, T>(&self, features: X, targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
        T: TryInto<SKTargetView<'a, f64>, Error = SKError>,
    {
        let features = features.try_into()?;
        let targets = targets.try_into()?;
        let training_rows = sk_number_of_rows(features);
        if training_rows != targets.len() {
            return Err(SKError::ShapeMismatch {
                expected: vec![training_rows],
                found: vec![targets.len()],
            });
        }
        Ok(SKMockThresholdClassifierModel {
            threshold: self.threshold,
            training_rows,
        })
    }
}

impl SKClassifierPredictor<f64> for SKMockThresholdClassifierModel {
    type Error = SKError;

    fn predict_labels<'a, X>(&self, features: X) -> Result<Array1<i64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
    {
        let features = features.try_into()?;
        let rows = sk_number_of_rows(features);
        Ok(Array1::from_iter((0..rows).map(|row| {
            i64::from(sk_first_column(features, row) >= self.threshold)
        })))
    }

    fn predict_probabilities<'a, X>(&self, features: X) -> Result<Array2<f64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
    {
        let labels = self.predict_labels(features)?;
        let mut probabilities = Array2::zeros((labels.len(), 2));
        for (row, label) in labels.iter().enumerate() {
            probabilities[[row, *label as usize]] = 1.0;
        }
        Ok(probabilities)
    }
}
