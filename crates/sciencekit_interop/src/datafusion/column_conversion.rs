//! Conversion of one DataFusion Arrow array into neutral column data (spec
//! `tabular-sources`).

use ::datafusion::arrow::array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array,
    Int64Array, LargeStringArray, StringArray, StringViewArray, UInt8Array, UInt16Array,
    UInt32Array, UInt64Array,
};
use ::datafusion::arrow::datatypes::DataType;
use sciencekit_common::SKError;

use crate::columns::SKColumnData;
use crate::schema::SKLogicalType;

macro_rules! sk_numeric_column {
    ($array:expr, $array_type:ty, $variant:ident) => {{
        let typed = $array
            .as_any()
            .downcast_ref::<$array_type>()
            .ok_or_else(|| SKError::SchemaMismatch {
                expected: format!("a {} array", stringify!($array_type)),
                found: format!("{:?}", $array.data_type()),
            })?;
        Ok(SKColumnData::$variant(
            (0..typed.len())
                .map(|row| {
                    if typed.is_null(row) {
                        None
                    } else {
                        Some(typed.value(row))
                    }
                })
                .collect(),
        ))
    }};
}

/// Convert one Arrow array into neutral, nullable column data.
///
/// # Errors
/// Returns [`SKError::SchemaMismatch`] when the concrete array does not match
/// the mapped logical type, and [`SKError::UnsupportedColumnType`] when a
/// `UInt64` value exceeds `i64`.
pub fn sk_column_data_from_datafusion_array(
    column_name: &str,
    array: &ArrayRef,
    logical_type: SKLogicalType,
) -> Result<SKColumnData, SKError> {
    match logical_type {
        SKLogicalType::Float32 => sk_numeric_column!(array, Float32Array, Float32),
        SKLogicalType::Float64 => sk_numeric_column!(array, Float64Array, Float64),
        SKLogicalType::Integer64 => sk_integer_column(column_name, array),
        SKLogicalType::Boolean => sk_numeric_column!(array, BooleanArray, Boolean),
        SKLogicalType::Nominal => sk_text_column(array),
        SKLogicalType::CategoryCode => Err(SKError::UnsupportedColumnType {
            column: column_name.to_string(),
            logical_type: logical_type.to_string(),
            suggestion: "the DataFusion adapter reads categories through their string labels"
                .to_string(),
        }),
    }
}

fn sk_integer_column(column_name: &str, array: &ArrayRef) -> Result<SKColumnData, SKError> {
    macro_rules! widened {
        ($array_type:ty) => {{
            let typed = array
                .as_any()
                .downcast_ref::<$array_type>()
                .ok_or_else(|| sk_array_mismatch::<$array_type>(array))?;
            (0..typed.len())
                .map(|row| {
                    if typed.is_null(row) {
                        None
                    } else {
                        Some(i64::from(typed.value(row)))
                    }
                })
                .collect::<Vec<_>>()
        }};
    }

    let values: Vec<Option<i64>> = match array.data_type() {
        DataType::Int8 => widened!(Int8Array),
        DataType::Int16 => widened!(Int16Array),
        DataType::Int32 => widened!(Int32Array),
        DataType::Int64 => widened!(Int64Array),
        DataType::UInt8 => widened!(UInt8Array),
        DataType::UInt16 => widened!(UInt16Array),
        DataType::UInt32 => widened!(UInt32Array),
        DataType::UInt64 => {
            let typed = array
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| sk_array_mismatch::<UInt64Array>(array))?;
            (0..typed.len())
                .map(|row| {
                    if typed.is_null(row) {
                        Ok(None)
                    } else {
                        i64::try_from(typed.value(row)).map(Some).map_err(|_| {
                            SKError::UnsupportedColumnType {
                                column: column_name.to_string(),
                                logical_type: "uint64".to_string(),
                                suggestion: "cast the column to int64 or float64 before import"
                                    .to_string(),
                            }
                        })
                    }
                })
                .collect::<Result<Vec<_>, SKError>>()?
        }
        other => {
            return Err(SKError::SchemaMismatch {
                expected: "an integer array".to_string(),
                found: format!("{other:?}"),
            });
        }
    };
    Ok(SKColumnData::Integer64(values))
}

fn sk_text_column(array: &ArrayRef) -> Result<SKColumnData, SKError> {
    match array.data_type() {
        DataType::Utf8 => {
            let typed = array
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| sk_array_mismatch::<StringArray>(array))?;
            Ok(SKColumnData::Nominal(
                (0..typed.len())
                    .map(|row| {
                        if typed.is_null(row) {
                            None
                        } else {
                            Some(typed.value(row).to_string())
                        }
                    })
                    .collect(),
            ))
        }
        DataType::LargeUtf8 => {
            let typed = array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .ok_or_else(|| sk_array_mismatch::<LargeStringArray>(array))?;
            Ok(SKColumnData::Nominal(
                (0..typed.len())
                    .map(|row| {
                        if typed.is_null(row) {
                            None
                        } else {
                            Some(typed.value(row).to_string())
                        }
                    })
                    .collect(),
            ))
        }
        DataType::Utf8View => {
            let typed = array
                .as_any()
                .downcast_ref::<StringViewArray>()
                .ok_or_else(|| sk_array_mismatch::<StringViewArray>(array))?;
            Ok(SKColumnData::Nominal(
                (0..typed.len())
                    .map(|row| {
                        if typed.is_null(row) {
                            None
                        } else {
                            Some(typed.value(row).to_string())
                        }
                    })
                    .collect(),
            ))
        }
        other => Err(SKError::SchemaMismatch {
            expected: "a text array".to_string(),
            found: format!("{other:?}"),
        }),
    }
}

fn sk_array_mismatch<T: 'static>(array: &ArrayRef) -> SKError {
    SKError::SchemaMismatch {
        expected: format!("a {} array", std::any::type_name::<T>()),
        found: format!("{:?}", array.data_type()),
    }
}
