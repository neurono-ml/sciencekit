//! Import of Arrow integer columns, including the lossless promotion required
//! by nulls under a floating-point policy (spec `arrow-columnar-interop`).

use arrow_array::{Array, ArrowPrimitiveType, PrimitiveArray};
use arrow_buffer::ArrowNativeType;
use sciencekit_common::SKError;

use super::null_mapping::sk_integer_null_fill;
use crate::columns::{SKColumnData, sk_integer_is_representable};
use crate::missing::SKMissingValuePolicy;

fn sk_integer_conversion_error(column_name: &str, value: &str) -> SKError {
    SKError::Conversion(format!(
        "column `{column_name}` carries the integer {value}, which does not fit the signed 64-bit neutral type"
    ))
}

fn sk_integer_to_i64<T>(value: T::Native, column_name: &str) -> Result<i64, SKError>
where
    T: ArrowPrimitiveType,
{
    value
        .to_i64()
        .ok_or_else(|| sk_integer_conversion_error(column_name, &format!("{value:?}")))
}

fn sk_integer_to_f64(value: i64, column_name: &str) -> Result<f64, SKError> {
    if !sk_integer_is_representable::<f64>(value) {
        return Err(SKError::Conversion(format!(
            "column `{column_name}` carries the integer {value}, which cannot be represented exactly after promotion to float64; cast the column to a floating type in the adapter before import"
        )));
    }
    Ok(value as f64)
}

/// Convert any Arrow integer column into a neutral column.
///
/// Complete columns stay [`SKColumnData::Integer64`]. A column carrying nulls
/// under [`SKMissingValuePolicy::NumericNaN`] or
/// [`SKMissingValuePolicy::ConstantFill`] is promoted to
/// [`SKColumnData::Float64`], because the neutral integer column has no null
/// placeholder; every value is checked for exact `f64` representation first so
/// the promotion never truncates.
///
/// # Errors
/// Returns [`SKError::NullValueInColumn`] under the reject policy and
/// [`SKError::Conversion`] for a value outside the neutral or promoted range.
pub(super) fn sk_integer_column<T>(
    name: &str,
    array: &PrimitiveArray<T>,
    policy: SKMissingValuePolicy,
) -> Result<SKColumnData, SKError>
where
    T: ArrowPrimitiveType,
    T::Native: ArrowNativeType,
{
    if array.null_count() == 0 {
        let values = array
            .iter()
            .map(|value| {
                value
                    .map(|value| sk_integer_to_i64::<T>(value, name).map(Some))
                    .unwrap_or(Ok(None))
            })
            .collect::<Result<Vec<_>, SKError>>()?;
        return Ok(SKColumnData::Integer64(values));
    }

    match policy {
        SKMissingValuePolicy::CategoricalMissingCode => Ok(SKColumnData::Integer64(
            array
                .iter()
                .map(|value| {
                    value
                        .map(|value| sk_integer_to_i64::<T>(value, name).map(Some))
                        .unwrap_or(Ok(None))
                })
                .collect::<Result<Vec<_>, SKError>>()?,
        )),
        SKMissingValuePolicy::Reject => Err(SKError::NullValueInColumn {
            column: name.to_string(),
            policy: policy.name(),
        }),
        SKMissingValuePolicy::NumericNaN | SKMissingValuePolicy::ConstantFill(_) => {
            let fill = sk_integer_null_fill(name, policy)?;
            let values = array
                .iter()
                .map(|value| match value {
                    Some(value) => {
                        let integer = sk_integer_to_i64::<T>(value, name)?;
                        sk_integer_to_f64(integer, name).map(Some)
                    }
                    None => Ok(Some(fill)),
                })
                .collect::<Result<Vec<_>, SKError>>()?;
            Ok(SKColumnData::Float64(values))
        }
    }
}
