//! Estimator families and math-backend names exposed to Lua.

use mlua::Value;

use crate::error::SKLuaError;

/// An estimator family the C ABI carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SKLuaEstimatorFamily {
    /// Ordinary least squares.
    LinearRegression,
    /// Column standardization.
    StandardScaler,
}

impl SKLuaEstimatorFamily {
    /// The stable family code the ABI expects.
    pub(crate) fn code(self) -> i32 {
        match self {
            SKLuaEstimatorFamily::LinearRegression => 0,
            SKLuaEstimatorFamily::StandardScaler => 1,
        }
    }

    /// The Lua-facing name, matching the constructor name.
    pub(crate) fn name(self) -> &'static str {
        match self {
            SKLuaEstimatorFamily::LinearRegression => "linear-regression",
            SKLuaEstimatorFamily::StandardScaler => "standard-scaler",
        }
    }
}

/// Translate a Lua backend value into the stable ABI backend code.
///
/// Strings are the documented form; integers are accepted so code that stores
/// the code directly keeps working.
pub(crate) fn sk_lua_backend_code(value: &Value) -> Result<i32, SKLuaError> {
    match value {
        Value::Integer(code) if (0..=2).contains(code) => Ok(*code as i32),
        Value::Integer(code) => Err(SKLuaError::invalid_argument(format!(
            "unknown backend code {code}; expected 0 (faer), 1 (matrix-multiply) or 2 (ndarray-linalg)"
        ))),
        Value::String(name) => {
            let name = name.to_str().map_err(SKLuaError::from)?;
            match name.as_ref() {
                "faer" => Ok(0),
                "matrix-multiply" | "matrixmultiply" => Ok(1),
                "ndarray-linalg" | "ndarray_linalg" => Ok(2),
                other => Err(SKLuaError::invalid_argument(format!(
                    "unknown backend {other:?}; expected \"faer\", \"matrix-multiply\" or \"ndarray-linalg\""
                ))),
            }
        }
        other => Err(SKLuaError::invalid_argument(format!(
            "the backend must be a string or an integer code, found {}",
            other.type_name()
        ))),
    }
}
