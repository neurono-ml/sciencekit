//! Runtime configuration erased behind one builder handle (spec `ffi-abi`).

use sciencekit_common::SKBackendKind;

use crate::array::SKFfiDataType;
use crate::error_kind::SKFfiError;

/// The estimator families the ABI exposes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKFfiEstimatorFamily {
    /// Ordinary least squares (`SKLinearRegression`).
    LinearRegression = 0,
    /// Standard scaler (`SKStandardScaler`).
    StandardScaler = 1,
}

impl SKFfiEstimatorFamily {
    /// Resolve the family code a binding sent.
    pub fn from_code(code: i32) -> Option<SKFfiEstimatorFamily> {
        match code {
            0 => Some(SKFfiEstimatorFamily::LinearRegression),
            1 => Some(SKFfiEstimatorFamily::StandardScaler),
            _ => None,
        }
    }
}

/// The configured, not-yet-fitted estimator behind an opaque handle.
#[derive(Debug, Clone)]
pub struct SKFfiBuilder {
    family: SKFfiEstimatorFamily,
    dtype: SKFfiDataType,
    include_intercept: bool,
    with_mean: bool,
    with_std: bool,
    backend: SKBackendKind,
}

impl SKFfiBuilder {
    /// A fresh builder with the Rust defaults for its family.
    pub fn new(family: SKFfiEstimatorFamily, dtype: SKFfiDataType) -> SKFfiBuilder {
        SKFfiBuilder {
            family,
            dtype,
            include_intercept: true,
            with_mean: true,
            with_std: true,
            backend: SKBackendKind::Faer,
        }
    }

    /// The configured family.
    pub fn family(&self) -> SKFfiEstimatorFamily {
        self.family
    }

    /// The configured scalar dtype.
    pub fn dtype(&self) -> SKFfiDataType {
        self.dtype
    }

    /// Whether the linear model solves for an intercept.
    pub fn include_intercept(&self) -> bool {
        self.include_intercept
    }

    /// Whether the scaler centers its columns.
    pub fn with_mean(&self) -> bool {
        self.with_mean
    }

    /// Whether the scaler divides by column standard deviation.
    pub fn with_std(&self) -> bool {
        self.with_std
    }

    /// The math backend kind.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// Set whether the linear model solves for an intercept.
    pub fn set_include_intercept(&mut self, enabled: bool) -> Result<(), SKFfiError> {
        self.require_family(SKFfiEstimatorFamily::LinearRegression, "include_intercept")?;
        self.include_intercept = enabled;
        Ok(())
    }

    /// Set whether the scaler centers its columns.
    pub fn set_with_mean(&mut self, enabled: bool) -> Result<(), SKFfiError> {
        self.require_family(SKFfiEstimatorFamily::StandardScaler, "with_mean")?;
        self.with_mean = enabled;
        Ok(())
    }

    /// Set whether the scaler divides by column standard deviation.
    pub fn set_with_std(&mut self, enabled: bool) -> Result<(), SKFfiError> {
        self.require_family(SKFfiEstimatorFamily::StandardScaler, "with_std")?;
        self.with_std = enabled;
        Ok(())
    }

    /// Set the math backend kind from its stable code.
    pub fn set_backend(&mut self, code: i32) -> Result<(), SKFfiError> {
        self.backend = match code {
            0 => SKBackendKind::Faer,
            1 => SKBackendKind::MatrixMultiply,
            2 => SKBackendKind::NdArrayLinalg,
            other => {
                return Err(SKFfiError::invalid_argument(format!(
                    "unknown backend code {other}; expected 0 (faer), 1 (matrixmultiply) or 2 (ndarray-linalg)"
                )));
            }
        };
        Ok(())
    }

    /// Reject a setter that belongs to another family.
    fn require_family(
        &self,
        expected: SKFfiEstimatorFamily,
        setter: &'static str,
    ) -> Result<(), SKFfiError> {
        if self.family == expected {
            return Ok(());
        }
        Err(SKFfiError::Unsupported {
            operation: setter,
            reason: format!(
                "the setter belongs to {:?}, but the builder configures {:?}",
                expected, self.family
            ),
        })
    }
}
