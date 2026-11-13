//! The fitted-model storage behind the opaque model handle (spec `ffi-abi`).

use sciencekit_common::SKFloat;
use sciencekit_linear_model::linear_regression::model::SKLinearRegressionModel;
use sciencekit_preprocessing::standard_scaler::model::SKStandardScalerModel;

use crate::array::SKFfiDataType;
use crate::builder::SKFfiEstimatorFamily;
use crate::error_kind::SKFfiError;

/// The fitted estimators a handle may carry in one scalar.
#[derive(Debug, Clone)]
pub enum SKFfiFitted<F: SKFloat> {
    /// A fitted ordinary-least-squares model.
    LinearRegression(SKLinearRegressionModel<F>),
    /// A fitted standard scaler.
    StandardScaler(SKStandardScalerModel<F>),
}

impl<F: SKFloat> SKFfiFitted<F> {
    /// The family of the fitted model.
    pub fn family(&self) -> SKFfiEstimatorFamily {
        match self {
            SKFfiFitted::LinearRegression(_) => SKFfiEstimatorFamily::LinearRegression,
            SKFfiFitted::StandardScaler(_) => SKFfiEstimatorFamily::StandardScaler,
        }
    }

    /// The scalar dtype of the fitted model.
    pub fn dtype(&self) -> SKFfiDataType {
        if size_of::<F>() == size_of::<f32>() {
            SKFfiDataType::F32
        } else {
            SKFfiDataType::F64
        }
    }

    /// The fitted linear-regression model, or an unsupported-operation error.
    pub fn linear_regression(&self) -> Result<&SKLinearRegressionModel<F>, SKFfiError> {
        match self {
            SKFfiFitted::LinearRegression(model) => Ok(model),
            SKFfiFitted::StandardScaler(_) => Err(SKFfiError::Unsupported {
                operation: "linear_regression",
                reason: "the handle carries a standard scaler".to_string(),
            }),
        }
    }

    /// The fitted standard scaler, or an unsupported-operation error.
    pub fn standard_scaler(&self) -> Result<&SKStandardScalerModel<F>, SKFfiError> {
        match self {
            SKFfiFitted::StandardScaler(model) => Ok(model),
            SKFfiFitted::LinearRegression(_) => Err(SKFfiError::Unsupported {
                operation: "standard_scaler",
                reason: "the handle carries a linear-regression model".to_string(),
            }),
        }
    }
}

/// A fitted model in a concrete scalar.
#[derive(Debug, Clone)]
pub enum SKFfiModelKind {
    /// Single-precision fitted models.
    F32(SKFfiFitted<f32>),
    /// Double-precision fitted models.
    F64(SKFfiFitted<f64>),
}

/// The opaque model handle payload.
#[derive(Debug, Clone)]
pub struct SKFfiModel {
    kind: SKFfiModelKind,
}

impl SKFfiModel {
    /// Wrap a double-precision fitted model.
    pub fn from_f64(fitted: SKFfiFitted<f64>) -> SKFfiModel {
        SKFfiModel {
            kind: SKFfiModelKind::F64(fitted),
        }
    }

    /// Wrap a single-precision fitted model.
    pub fn from_f32(fitted: SKFfiFitted<f32>) -> SKFfiModel {
        SKFfiModel {
            kind: SKFfiModelKind::F32(fitted),
        }
    }

    /// The scalar dtype of the stored model.
    pub fn dtype(&self) -> SKFfiDataType {
        match &self.kind {
            SKFfiModelKind::F32(_) => SKFfiDataType::F32,
            SKFfiModelKind::F64(_) => SKFfiDataType::F64,
        }
    }

    /// The family of the stored model.
    pub fn family(&self) -> SKFfiEstimatorFamily {
        match &self.kind {
            SKFfiModelKind::F32(fitted) => fitted.family(),
            SKFfiModelKind::F64(fitted) => fitted.family(),
        }
    }

    /// The stored model kind.
    pub fn kind(&self) -> &SKFfiModelKind {
        &self.kind
    }
}
