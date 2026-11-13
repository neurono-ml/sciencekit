//! The frozen base a low-rank adapter is attached to (spec `lora-adapters`).

use ndarray::{Array1, Array2, Axis};
use sciencekit_common::{SKError, SKFloat};

use crate::linear_regression::model::SKLinearRegressionModel;
use crate::logistic_regression::model::SKLogisticRegressionModel;

use super::config::SKLoraTargetModule;

/// A fitted model whose parameter matrix an adapter may correct.
///
/// The adapter borrows the base through this boundary, never through a mutable
/// reference: the base parameters stay untouched for the whole adapter
/// lifetime. The initial coverage names the two matrix-parameter families; any
/// other family enters as [`SKLoraBaseModel::UnsupportedFamily`] so attaching
/// fails with a precise [`SKError::UnsupportedOperation`] instead of a panic.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SKLoraBaseModel<F> {
    /// A fitted linear-regression model (a `1 × features` coefficient block).
    LinearRegression(SKLinearRegressionModel<F>),
    /// A fitted logistic-regression model (a `heads × features` block).
    LogisticRegression(SKLogisticRegressionModel<F>),
    /// A base family with no adaptable coefficient block (trees, neighbours).
    UnsupportedFamily {
        /// The family name reported by the rejection.
        family: &'static str,
    },
}

impl<F: SKFloat> SKLoraBaseModel<F> {
    /// Wrap a fitted linear-regression model.
    pub fn linear_regression(model: SKLinearRegressionModel<F>) -> SKLoraBaseModel<F> {
        SKLoraBaseModel::LinearRegression(model)
    }

    /// Wrap a fitted logistic-regression model.
    pub fn logistic_regression(model: SKLogisticRegressionModel<F>) -> SKLoraBaseModel<F> {
        SKLoraBaseModel::LogisticRegression(model)
    }

    /// Name a base family this adapter cannot cover.
    ///
    /// Attaching an adapter to it returns [`SKError::UnsupportedOperation`],
    /// naming this family in the reason.
    pub fn unsupported_family(family: &'static str) -> SKLoraBaseModel<F> {
        SKLoraBaseModel::UnsupportedFamily { family }
    }

    /// The base family name (`linear_regression`, `logistic_regression` or the
    /// name carried by an unsupported family).
    pub fn family(&self) -> &'static str {
        match self {
            SKLoraBaseModel::LinearRegression(_) => "linear_regression",
            SKLoraBaseModel::LogisticRegression(_) => "logistic_regression",
            SKLoraBaseModel::UnsupportedFamily { family } => family,
        }
    }

    /// The number of features the base was fitted on.
    pub fn feature_count(&self) -> usize {
        match self {
            SKLoraBaseModel::LinearRegression(model) => model.coefficients().len(),
            SKLoraBaseModel::LogisticRegression(model) => model.feature_count(),
            SKLoraBaseModel::UnsupportedFamily { .. } => 0,
        }
    }

    /// The target module this family exposes, when it exposes one.
    pub fn target_module(&self) -> Option<SKLoraTargetModule> {
        match self {
            SKLoraBaseModel::LinearRegression(_) => Some(SKLoraTargetModule::LinearCoefficients),
            SKLoraBaseModel::LogisticRegression(_) => {
                Some(SKLoraTargetModule::LogisticCoefficients)
            }
            SKLoraBaseModel::UnsupportedFamily { .. } => None,
        }
    }

    /// The rejection an adapter gets for this base family.
    pub(crate) fn unsupported_error(&self) -> SKError {
        SKError::UnsupportedOperation {
            operation: "attach",
            reason: format!(
                "the `{}` family has no adaptable coefficient matrix; supported families are \
                 linear regression and logistic regression",
                self.family()
            ),
        }
    }

    /// The adapted parameter block, shaped `heads × features`.
    pub(crate) fn coefficients(&self) -> Result<Array2<F>, SKError> {
        match self {
            SKLoraBaseModel::LinearRegression(model) => {
                Ok(model.coefficients().view().insert_axis(Axis(0)).to_owned())
            }
            SKLoraBaseModel::LogisticRegression(model) => Ok(model.coefficients().to_owned()),
            SKLoraBaseModel::UnsupportedFamily { .. } => Err(self.unsupported_error()),
        }
    }

    /// The per-head intercepts, one per row of [`Self::coefficients`].
    pub(crate) fn intercepts(&self) -> Result<Array1<F>, SKError> {
        match self {
            SKLoraBaseModel::LinearRegression(model) => Ok(Array1::from_elem(1, model.intercept())),
            SKLoraBaseModel::LogisticRegression(model) => Ok(model.intercepts().to_owned()),
            SKLoraBaseModel::UnsupportedFamily { .. } => Err(self.unsupported_error()),
        }
    }
}
