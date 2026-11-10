//! Deterministic estimators used by the grid-search TDD suites.
//!
//! They are intentionally tiny: a constant regressor whose response is the
//! configured hyperparameter, a threshold classifier whose decision boundary is
//! the configured hyperparameter, a variant that rejects dense features (to
//! prove the grid never densifies) and fixed-output models for hand-computed
//! scoring checks.

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKClassifierPredictor, SKDataView, SKError, SKRegressorPredictor, SKSupervisedFit, SKTargetView,
};

fn sk_number_of_rows(features: SKDataView<'_, f64>) -> usize {
    match features {
        SKDataView::Dense(view) => view.nrows(),
        SKDataView::Sparse(view) => view.rows(),
        _ => 0,
    }
}

fn sk_first_column(features: SKDataView<'_, f64>, row: usize) -> f64 {
    match features {
        SKDataView::Dense(view) => view[[row, 0]],
        SKDataView::Sparse(view) => view.get(row, 0).copied().unwrap_or(0.0),
        _ => 0.0,
    }
}

/// A regressor predicting the constant configured response.
#[derive(Debug, Clone)]
pub(crate) struct SKMockConstantRegressor {
    value: f64,
    fail_when_target_present: Option<f64>,
}

impl SKMockConstantRegressor {
    pub(crate) fn new(value: f64) -> Self {
        Self {
            value,
            fail_when_target_present: None,
        }
    }

    pub(crate) fn failing_when_target_present(mut self, marker: f64) -> Self {
        self.fail_when_target_present = Some(marker);
        self
    }
}

/// The fitted constant regressor.
#[derive(Debug, Clone)]
pub(crate) struct SKMockConstantRegressorModel {
    value: f64,
    training_rows: usize,
    sparse_training: bool,
}

impl SKMockConstantRegressorModel {
    pub(crate) fn training_rows(&self) -> usize {
        self.training_rows
    }

    pub(crate) fn sparse_training(&self) -> bool {
        self.sparse_training
    }
}

impl SKSupervisedFit<f64> for SKMockConstantRegressor {
    type Model = SKMockConstantRegressorModel;
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
        if let Some(marker) = self.fail_when_target_present {
            let values = targets.as_continuous()?;
            if values.iter().any(|value| *value == marker) {
                return Err(SKError::NotConverged { iterations: 7 });
            }
        }
        Ok(SKMockConstantRegressorModel {
            value: self.value,
            training_rows,
            sparse_training: matches!(features, SKDataView::Sparse(_)),
        })
    }
}

impl SKRegressorPredictor<f64> for SKMockConstantRegressorModel {
    type Error = SKError;

    fn predict<'a, X>(&self, features: X) -> Result<Array1<f64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
    {
        let features = features.try_into()?;
        Ok(Array1::from_elem(sk_number_of_rows(features), self.value))
    }
}

/// A regressor rejecting dense features: usable only when the grid preserves
/// the sparse representation end to end.
#[derive(Debug, Clone, Default)]
pub(crate) struct SKMockSparseOnlyRegressor;

impl SKSupervisedFit<f64> for SKMockSparseOnlyRegressor {
    type Model = SKMockSparseOnlyRegressorModel;
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
        match features {
            SKDataView::Sparse(_) => Ok(SKMockSparseOnlyRegressorModel {
                training_rows,
                value: 0.0,
            }),
            SKDataView::Dense(_) => Err(SKError::UnsupportedRepresentation {
                representation: "dense",
                suggestion: "this mock only accepts CSR folds",
            }),
            _ => Err(SKError::UnsupportedRepresentation {
                representation: "unknown",
                suggestion: "this mock only accepts CSR folds",
            }),
        }
    }
}

/// The fitted sparse-only regressor.
#[derive(Debug, Clone)]
pub(crate) struct SKMockSparseOnlyRegressorModel {
    training_rows: usize,
    value: f64,
}

impl SKMockSparseOnlyRegressorModel {
    pub(crate) fn training_rows(&self) -> usize {
        self.training_rows
    }
}

impl SKRegressorPredictor<f64> for SKMockSparseOnlyRegressorModel {
    type Error = SKError;

    fn predict<'a, X>(&self, features: X) -> Result<Array1<f64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, f64>, Error = SKError>,
    {
        let features = features.try_into()?;
        match features {
            SKDataView::Sparse(view) => Ok(Array1::from_elem(view.rows(), self.value)),
            SKDataView::Dense(_) => Err(SKError::UnsupportedRepresentation {
                representation: "dense",
                suggestion: "this mock only accepts CSR folds",
            }),
            _ => Err(SKError::UnsupportedRepresentation {
                representation: "unknown",
                suggestion: "this mock only accepts CSR folds",
            }),
        }
    }
}

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

/// A classifier model with fixed outputs, used for hand-computed metric checks.
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

/// A regressor model with fixed predictions, used for hand-computed checks.
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
