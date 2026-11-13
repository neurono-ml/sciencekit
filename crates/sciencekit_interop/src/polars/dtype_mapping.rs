//! Mapping of Polars data types onto the neutral logical vocabulary (spec
//! `tabular-sources`).

use ::polars::prelude::DataType;
use sciencekit_common::SKError;

use crate::schema::SKLogicalType;

/// Map one Polars data type onto a neutral logical type.
///
/// Signed and unsigned integers up to 32 bits widen losslessly into
/// [`SKLogicalType::Integer64`]; `UInt64` also maps to `Integer64` but its
/// values are range-checked during row conversion. Categorical and Enum
/// columns are read through their string labels, so they become
/// [`SKLogicalType::Nominal`].
///
/// # Errors
/// Returns [`SKError::UnsupportedColumnType`] naming the column for temporal,
/// nested, binary, decimal, 128-bit and all-null columns.
pub fn sk_logical_type_from_polars_data_type(
    data_type: &DataType,
    column_name: &str,
) -> Result<SKLogicalType, SKError> {
    match data_type {
        DataType::Boolean => Ok(SKLogicalType::Boolean),
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => Ok(SKLogicalType::Integer64),
        DataType::Float32 => Ok(SKLogicalType::Float32),
        DataType::Float64 => Ok(SKLogicalType::Float64),
        DataType::String => Ok(SKLogicalType::Nominal),
        DataType::Categorical(_, _) | DataType::Enum(_, _) => Ok(SKLogicalType::Nominal),
        DataType::Null => Err(sk_unsupported(
            column_name,
            "null",
            "drop all-null columns or cast them to a concrete type before import",
        )),
        DataType::Date | DataType::Datetime(_, _) | DataType::Duration(_) | DataType::Time => {
            Err(sk_unsupported(
                column_name,
                "temporal",
                "convert the temporal column to a numeric epoch column before import",
            ))
        }
        DataType::List(_) => Err(sk_unsupported(
            column_name,
            "nested",
            "flatten the nested column into scalar columns before import",
        )),
        DataType::Binary | DataType::BinaryOffset => Err(sk_unsupported(
            column_name,
            "binary",
            "cast the binary column to text before import",
        )),
        DataType::Int128 => Err(sk_unsupported(
            column_name,
            "int128",
            "cast the column to int64 or float64 before import",
        )),
        other => Err(sk_unsupported(
            column_name,
            &format!("{other:?}"),
            "convert the column to float32, float64, integer64, boolean or text before import",
        )),
    }
}

fn sk_unsupported(column_name: &str, family: &str, suggestion: &str) -> SKError {
    SKError::UnsupportedColumnType {
        column: column_name.to_string(),
        logical_type: family.to_string(),
        suggestion: suggestion.to_string(),
    }
}
