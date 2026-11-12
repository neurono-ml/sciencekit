//! Import of Arrow floating-point columns (spec `arrow-columnar-interop`).

use arrow_array::PrimitiveArray;
use arrow_array::types::{Float32Type, Float64Type};
use sciencekit_common::SKError;

use super::null_mapping::{sk_float_for_null, sk_float32_from_fill};
use crate::columns::SKColumnData;
use crate::missing::SKMissingValuePolicy;

/// Convert a `Float32` Arrow column, applying the missing-value policy to null
/// positions.
///
/// # Errors
/// Returns [`SKError::NullValueInColumn`] under the reject policy and
/// [`SKError::InvalidHyperparameter`] when a fill value overflows `f32`.
pub(super) fn sk_float32_column(
    name: &str,
    array: &PrimitiveArray<Float32Type>,
    policy: SKMissingValuePolicy,
) -> Result<SKColumnData, SKError> {
    let values = array
        .iter()
        .map(|value| match value {
            Some(value) => Ok(Some(value)),
            None => match sk_float_for_null(name, policy)? {
                Some(fill) => Ok(Some(sk_float32_from_fill(name, fill)?)),
                None => Ok(None),
            },
        })
        .collect::<Result<Vec<_>, SKError>>()?;
    Ok(SKColumnData::Float32(values))
}

/// Convert a `Float64` Arrow column, applying the missing-value policy to null
/// positions.
///
/// # Errors
/// Returns [`SKError::NullValueInColumn`] under the reject policy.
pub(super) fn sk_float64_column(
    name: &str,
    array: &PrimitiveArray<Float64Type>,
    policy: SKMissingValuePolicy,
) -> Result<SKColumnData, SKError> {
    let values = array
        .iter()
        .map(|value| match value {
            Some(value) => Ok(Some(value)),
            None => sk_float_for_null(name, policy),
        })
        .collect::<Result<Vec<_>, SKError>>()?;
    Ok(SKColumnData::Float64(values))
}
