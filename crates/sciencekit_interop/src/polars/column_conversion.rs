//! Conversion of one Polars series into neutral column data (spec
//! `tabular-sources`).

use ::polars::prelude::{AnyValue, DataType, Series};
use sciencekit_common::SKError;

use crate::columns::SKColumnData;
use crate::schema::SKLogicalType;

/// Convert one series into neutral, nullable column data.
///
/// The series is first cast to its neutral representation when Polars stores it
/// behind a logical type (categorical and enum columns become strings), so the
/// row loop only ever sees the physical values the neutral vocabulary carries.
///
/// # Errors
/// Returns [`SKError::Conversion`] for a Polars failure and
/// [`SKError::UnsupportedColumnType`] when a `UInt64` value exceeds `i64`.
pub fn sk_column_data_from_polars_series(
    column_name: &str,
    series: &Series,
    logical_type: SKLogicalType,
) -> Result<SKColumnData, SKError> {
    let series = match logical_type {
        SKLogicalType::Nominal => series.cast(&DataType::String).map_err(|error| {
            SKError::Conversion(format!(
                "casting column `{column_name}` to text failed: {error}"
            ))
        })?,
        _ => series.clone(),
    };

    match logical_type {
        SKLogicalType::Float32 => Ok(SKColumnData::Float32(
            series
                .iter()
                .map(|value| match value {
                    AnyValue::Float32(value) => Some(value),
                    _ => None,
                })
                .collect(),
        )),
        SKLogicalType::Float64 => Ok(SKColumnData::Float64(
            series
                .iter()
                .map(|value| match value {
                    AnyValue::Float64(value) => Some(value),
                    _ => None,
                })
                .collect(),
        )),
        SKLogicalType::Integer64 => series
            .iter()
            .map(|value| sk_integer_value(column_name, value))
            .collect::<Result<Vec<_>, SKError>>()
            .map(SKColumnData::Integer64),
        SKLogicalType::Boolean => Ok(SKColumnData::Boolean(
            series
                .iter()
                .map(|value| match value {
                    AnyValue::Boolean(value) => Some(value),
                    _ => None,
                })
                .collect(),
        )),
        SKLogicalType::Nominal => Ok(SKColumnData::Nominal(
            series
                .iter()
                .map(|value| match value {
                    AnyValue::String(value) => Some(value.to_string()),
                    AnyValue::StringOwned(value) => Some(value.to_string()),
                    _ => None,
                })
                .collect(),
        )),
        SKLogicalType::CategoryCode => Err(SKError::UnsupportedColumnType {
            column: column_name.to_string(),
            logical_type: logical_type.to_string(),
            suggestion: "the Polars adapter reads categories through their string labels"
                .to_string(),
        }),
    }
}

fn sk_integer_value(column_name: &str, value: AnyValue<'_>) -> Result<Option<i64>, SKError> {
    let integer = match value {
        AnyValue::Int8(value) => i64::from(value),
        AnyValue::Int16(value) => i64::from(value),
        AnyValue::Int32(value) => i64::from(value),
        AnyValue::Int64(value) => value,
        AnyValue::UInt8(value) => i64::from(value),
        AnyValue::UInt16(value) => i64::from(value),
        AnyValue::UInt32(value) => i64::from(value),
        AnyValue::UInt64(value) => {
            i64::try_from(value).map_err(|_| SKError::UnsupportedColumnType {
                column: column_name.to_string(),
                logical_type: "uint64".to_string(),
                suggestion: "cast the column to int64 or float64 before import".to_string(),
            })?
        }
        _ => return Ok(None),
    };
    Ok(Some(integer))
}
