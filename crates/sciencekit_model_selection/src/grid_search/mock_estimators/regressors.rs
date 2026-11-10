//! Constant and sparse-only mock regressors.

use ndarray::Array1;
use sciencekit_common::{SKDataView, SKError, SKRegressorPredictor, SKSupervisedFit, SKTargetView};

use super::helpers::sk_number_of_rows;

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

    pub(crate) fn sparse_training(&self) -> bool {
        true
    }
}

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
