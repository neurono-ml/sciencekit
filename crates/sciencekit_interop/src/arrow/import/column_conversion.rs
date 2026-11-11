//! Dispatch of one Arrow array onto its neutral column representation (spec
//! `arrow-columnar-interop`).

use arrow_array::types::{
    Float32Type, Float64Type, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type,
    UInt32Type,
};
use arrow_array::{Array, PrimitiveArray};
use arrow_schema::DataType;
use sciencekit_common::SKError;

use super::boolean_conversion::sk_boolean_column;
use super::dictionary_conversion::sk_dictionary_column;
use super::float_conversion::{sk_float32_column, sk_float64_column};
use super::integer_conversion::sk_integer_column;
use super::text_conversion::{sk_large_string_column, sk_string_column, sk_string_view_column};
use crate::columns::SKColumnData;
use crate::missing::SKMissingValuePolicy;

fn sk_array_conversion_error(column_name: &str, data_type: &DataType) -> SKError {
    SKError::UnsupportedColumnType {
        column: column_name.to_string(),
        logical_type: data_type.to_string(),
        suggestion:
            "convert the column to float32, float64, integer64, boolean or text before import"
                .to_string(),
    }
}

macro_rules! sk_primitive_integer_column {
    ($array:expr, $ty:ty, $name:expr, $policy:expr) => {{
        let typed = $array
            .as_any()
            .downcast_ref::<PrimitiveArray<$ty>>()
            .expect("the data type and the downcast agree");
        sk_integer_column::<$ty>($name, typed, $policy)
    }};
}

/// Convert one Arrow array into neutral column data under the declared policy.
///
/// Arrow validity bitmaps are rejected immediately under
/// [`SKMissingValuePolicy::Reject`]. Floating-point columns materialize
/// [`SKMissingValuePolicy::NumericNaN`] and
/// [`SKMissingValuePolicy::ConstantFill`] into the floating buffer at import,
/// while integer, boolean, text and dictionary columns retain `None` at null
/// positions so downstream numeric, categorical and target extractions apply
/// scalar precision guards and sentinel codes without mutating the batch
/// schema.
///
/// # Errors
/// Returns [`SKError::UnsupportedColumnType`] for a type the schema mapping
/// should already have rejected, and propagates the policy and conversion
/// errors of the concrete column importers.
pub fn sk_column_data_from_arrow_array(
    column_name: &str,
    array: &dyn Array,
    policy: SKMissingValuePolicy,
) -> Result<SKColumnData, SKError> {
    match array.data_type() {
        DataType::Float32 => Ok(sk_float32_column(
            column_name,
            array
                .as_any()
                .downcast_ref::<PrimitiveArray<Float32Type>>()
                .expect("the data type and the downcast agree"),
            policy,
        )?),
        DataType::Float64 => sk_float64_column(
            column_name,
            array
                .as_any()
                .downcast_ref::<PrimitiveArray<Float64Type>>()
                .expect("the data type and the downcast agree"),
            policy,
        ),
        DataType::Int8 => sk_primitive_integer_column!(array, Int8Type, column_name, policy),
        DataType::Int16 => sk_primitive_integer_column!(array, Int16Type, column_name, policy),
        DataType::Int32 => sk_primitive_integer_column!(array, Int32Type, column_name, policy),
        DataType::Int64 => sk_primitive_integer_column!(array, Int64Type, column_name, policy),
        DataType::UInt8 => sk_primitive_integer_column!(array, UInt8Type, column_name, policy),
        DataType::UInt16 => sk_primitive_integer_column!(array, UInt16Type, column_name, policy),
        DataType::UInt32 => sk_primitive_integer_column!(array, UInt32Type, column_name, policy),
        DataType::Boolean => sk_boolean_column(
            column_name,
            array
                .as_any()
                .downcast_ref::<arrow_array::BooleanArray>()
                .expect("the data type and the downcast agree"),
            policy,
        ),
        DataType::Utf8 => sk_string_column(
            column_name,
            array
                .as_any()
                .downcast_ref::<arrow_array::StringArray>()
                .expect("the data type and the downcast agree"),
            policy,
        ),
        DataType::LargeUtf8 => sk_large_string_column(
            column_name,
            array
                .as_any()
                .downcast_ref::<arrow_array::LargeStringArray>()
                .expect("the data type and the downcast agree"),
            policy,
        ),
        DataType::Utf8View => sk_string_view_column(
            column_name,
            array
                .as_any()
                .downcast_ref::<arrow_array::StringViewArray>()
                .expect("the data type and the downcast agree"),
            policy,
        ),
        DataType::Dictionary(_, _) => sk_dictionary_column(column_name, array, policy),
        other => Err(sk_array_conversion_error(column_name, other)),
    }
}
