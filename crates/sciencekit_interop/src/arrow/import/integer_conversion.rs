//! Import of Arrow integer columns into the neutral 64-bit integer column
//! representation (spec `arrow-columnar-interop`).

use arrow_array::{Array, ArrowPrimitiveType, PrimitiveArray};
use arrow_buffer::ArrowNativeType;
use sciencekit_common::SKError;

use super::null_mapping::sk_validate_constant_fill_policy;
use crate::columns::SKColumnData;
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

/// Convert any supported Arrow integer column into [`SKColumnData::Integer64`].
///
/// Both complete and nullable columns remain [`SKColumnData::Integer64`] so the
/// batch schema never mutates across a stream and exact `i64` magnitudes are
/// checked against the concrete model scalar (`f32` or `f64`) at
/// materialization time rather than truncated prematurely.
///
/// # Errors
/// Returns [`SKError::NullValueInColumn`] when `policy` is `Reject` and the
/// column carries nulls, [`SKError::InvalidHyperparameter`] for a non-finite
/// constant fill, and [`SKError::Conversion`] for a value outside signed
/// 64-bit range.
pub(super) fn sk_integer_column<T>(
    name: &str,
    array: &PrimitiveArray<T>,
    policy: SKMissingValuePolicy,
) -> Result<SKColumnData, SKError>
where
    T: ArrowPrimitiveType,
    T::Native: ArrowNativeType,
{
    sk_validate_constant_fill_policy(name, policy)?;
    if policy.is_reject() && array.null_count() > 0 {
        return Err(SKError::NullValueInColumn {
            column: name.to_string(),
            policy: policy.name(),
        });
    }

    let values = array
        .iter()
        .map(|value| match value {
            Some(value) => sk_integer_to_i64::<T>(value, name).map(Some),
            None => Ok(None),
        })
        .collect::<Result<Vec<_>, SKError>>()?;
    Ok(SKColumnData::Integer64(values))
}
