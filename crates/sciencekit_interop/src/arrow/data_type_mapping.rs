//! Mapping of Arrow data types onto the neutral logical vocabulary (spec
//! `arrow-columnar-interop`).
//!
//! The mapping is deliberately narrow: only the types the numeric and
//! categorical paths can represent cross the seam. Everything else is rejected
//! up front with a structured error that names the Arrow field and suggests a
//! conversion, so a nested or temporal column never reaches materialization.

use arrow_schema::DataType;
use sciencekit_common::SKError;

use crate::schema::SKLogicalType;

/// Build the structured rejection for a type family the seam does not carry.
fn sk_unsupported_type(field_name: &str, family: &str, suggestion: &str) -> SKError {
    SKError::UnsupportedColumnType {
        column: field_name.to_string(),
        logical_type: family.to_string(),
        suggestion: suggestion.to_string(),
    }
}

/// Map one Arrow data type onto a neutral logical type.
///
/// Signed and unsigned integers up to 32 bits widen losslessly into
/// [`SKLogicalType::Integer64`]; `UInt64` is rejected because it can exceed the
/// signed 64-bit range. String dictionaries become
/// [`SKLogicalType::CategoryCode`] because the dictionary values are the label
/// table; plain strings stay [`SKLogicalType::Nominal`].
///
/// # Errors
/// Returns [`SKError::UnsupportedColumnType`] naming `field_name` for nested,
/// temporal, binary, decimal, `Float16`, `UInt64` and `Null` columns.
pub fn sk_logical_type_from_arrow_data_type(
    data_type: &DataType,
    field_name: &str,
) -> Result<SKLogicalType, SKError> {
    match data_type {
        DataType::Boolean => Ok(SKLogicalType::Boolean),
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
            Ok(SKLogicalType::Integer64)
        }
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 => Ok(SKLogicalType::Integer64),
        DataType::Float32 => Ok(SKLogicalType::Float32),
        DataType::Float64 => Ok(SKLogicalType::Float64),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => Ok(SKLogicalType::Nominal),
        DataType::Dictionary(key, value)
            if key.is_integer() && matches!(**value, DataType::Utf8 | DataType::LargeUtf8) =>
        {
            Ok(SKLogicalType::CategoryCode)
        }
        DataType::Dictionary(_, _) => Err(sk_unsupported_type(
            field_name,
            "dictionary",
            "encode the dictionary with string values in the adapter before import",
        )),
        DataType::Null => Err(sk_unsupported_type(
            field_name,
            "null",
            "drop all-null columns or cast them to a concrete type in the adapter before import",
        )),
        DataType::UInt64 => Err(sk_unsupported_type(
            field_name,
            "unsigned_64",
            "cast the column to int64 or float64 in the adapter before import",
        )),
        DataType::Float16 => Err(sk_unsupported_type(
            field_name,
            "float16",
            "cast the column to float32 in the adapter before import",
        )),
        DataType::Timestamp(_, _)
        | DataType::Date32
        | DataType::Date64
        | DataType::Time32(_)
        | DataType::Time64(_)
        | DataType::Duration(_)
        | DataType::Interval(_) => Err(sk_unsupported_type(
            field_name,
            "temporal",
            "convert the temporal column to a numeric epoch column in the adapter before import",
        )),
        DataType::Binary
        | DataType::LargeBinary
        | DataType::BinaryView
        | DataType::FixedSizeBinary(_) => Err(sk_unsupported_type(
            field_name,
            "binary",
            "cast the binary column to text in the adapter before import",
        )),
        DataType::Decimal128(_, _) | DataType::Decimal256(_, _) => Err(sk_unsupported_type(
            field_name,
            "decimal",
            "cast the decimal column to float64 in the adapter before import",
        )),
        DataType::List(_)
        | DataType::ListView(_)
        | DataType::FixedSizeList(_, _)
        | DataType::LargeList(_)
        | DataType::LargeListView(_) => Err(sk_unsupported_type(
            field_name,
            "nested",
            "flatten the list column into scalar columns in the adapter before import",
        )),
        DataType::Struct(_) | DataType::Union(_, _) | DataType::Map(_, _) => {
            Err(sk_unsupported_type(
                field_name,
                "nested",
                "flatten the nested column into scalar columns in the adapter before import",
            ))
        }
        _ => Err(sk_unsupported_type(
            field_name,
            "unsupported",
            "convert the column to float32, float64, integer64, boolean or text before import",
        )),
    }
}
