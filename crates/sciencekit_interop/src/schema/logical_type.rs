//! The closed set of logical column types the neutral seam understands
//! (spec `dataset-interop-core`).

use std::fmt;

use sciencekit_common::SKError;

/// The logical type carried by a column, independent of the engine that
/// produced it.
///
/// The set is closed and deliberately small: it is exactly the vocabulary the
/// numeric materialization and the categorical/target extraction paths support.
/// Engine adapters map their richer type systems onto it and reject anything
/// that has no neutral counterpart before the batch reaches this crate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SKLogicalType {
    /// Single-precision floating-point values.
    Float32,
    /// Double-precision floating-point values.
    Float64,
    /// Signed 64-bit integer values.
    Integer64,
    /// Boolean values, coerced to `0`/`1` on the numeric path.
    Boolean,
    /// Nominal (text) values.
    Nominal,
    /// Integer category codes, including the documented missing sentinel.
    CategoryCode,
}

impl SKLogicalType {
    /// The stable, lower-case name reported in errors, traces and metadata.
    pub fn name(&self) -> &'static str {
        match self {
            SKLogicalType::Float32 => "float32",
            SKLogicalType::Float64 => "float64",
            SKLogicalType::Integer64 => "integer64",
            SKLogicalType::Boolean => "boolean",
            SKLogicalType::Nominal => "nominal",
            SKLogicalType::CategoryCode => "category_code",
        }
    }

    /// Parse a logical type from its stable name.
    ///
    /// Rejects an unknown name with the central taxonomy instead of silently
    /// falling back to a default, so a malformed adapter mapping is visible.
    pub fn from_name(name: &str) -> Result<Self, SKError> {
        match name {
            "float32" => Ok(SKLogicalType::Float32),
            "float64" => Ok(SKLogicalType::Float64),
            "integer64" => Ok(SKLogicalType::Integer64),
            "boolean" => Ok(SKLogicalType::Boolean),
            "nominal" => Ok(SKLogicalType::Nominal),
            "category_code" => Ok(SKLogicalType::CategoryCode),
            other => Err(SKError::Conversion(format!(
                "unknown logical column type `{other}`; expected one of float32, float64, integer64, boolean, nominal, category_code"
            ))),
        }
    }

    /// Whether the type enters the numeric feature matrix without an explicit
    /// categorical encoding.
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            SKLogicalType::Float32
                | SKLogicalType::Float64
                | SKLogicalType::Integer64
                | SKLogicalType::Boolean
        )
    }

    /// Whether the type enters the categorical view.
    pub fn is_categorical(&self) -> bool {
        matches!(self, SKLogicalType::Nominal | SKLogicalType::CategoryCode)
    }
}

impl fmt::Display for SKLogicalType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}
