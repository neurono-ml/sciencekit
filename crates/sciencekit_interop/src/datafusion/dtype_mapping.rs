//! Mapping of DataFusion's Arrow data types onto the neutral vocabulary (spec
//! `tabular-sources`).

use ::datafusion::arrow::datatypes::DataType;
use sciencekit_common::SKError;

use crate::schema::SKLogicalType;

/// Map one Arrow data type onto a neutral logical type.
///
/// DataFusion 46 carries Arrow 54, so it has its own type enum — separate from
/// the workspace's Arrow 58 adapter. Only the types the numeric and categorical
/// paths represent cross the seam.
///
/// # Errors
/// Returns [`SKError::UnsupportedColumnType`] naming the column for nested,
/// temporal, binary, decimal, 128-bit and all-null columns.
pub fn sk_logical_type_from_datafusion_data_type(
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
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => Ok(SKLogicalType::Nominal),
        DataType::Null => Err(sk_unsupported(
            column_name,
            "null",
            "drop all-null columns or cast them to a concrete type before import",
        )),
        DataType::Timestamp(_, _)
        | DataType::Date32
        | DataType::Date64
        | DataType::Time32(_)
        | DataType::Time64(_)
        | DataType::Duration(_)
        | DataType::Interval(_) => Err(sk_unsupported(
            column_name,
            "temporal",
            "convert the temporal column to a numeric epoch column before import",
        )),
        DataType::List(_) | DataType::LargeList(_) | DataType::FixedSizeList(_, _) => {
            Err(sk_unsupported(
                column_name,
                "nested",
                "flatten the nested column into scalar columns before import",
            ))
        }
        DataType::Struct(_) | DataType::Union(_, _) | DataType::Map(_, _) => Err(sk_unsupported(
            column_name,
            "nested",
            "flatten the nested column into scalar columns before import",
        )),
        DataType::Binary | DataType::LargeBinary | DataType::BinaryView => Err(sk_unsupported(
            column_name,
            "binary",
            "cast the binary column to text before import",
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
