//! Import of Arrow boolean columns (spec `arrow-columnar-interop`).

use arrow_array::{Array, BooleanArray};
use sciencekit_common::SKError;

use crate::columns::SKColumnData;
use crate::missing::SKMissingValuePolicy;

/// Convert a `Boolean` Arrow column.
///
/// A complete column becomes [`SKColumnData::Boolean`], which the numeric path
/// coerces to `0`/`1`. A null under the reject policy fails; under any other
/// policy the null is retained for the downstream conversion, which either
/// writes the numeric sentinel or the categorical missing code.
///
/// # Errors
/// Returns [`SKError::NullValueInColumn`] under the reject policy.
pub(super) fn sk_boolean_column(
    name: &str,
    array: &BooleanArray,
    policy: SKMissingValuePolicy,
) -> Result<SKColumnData, SKError> {
    if policy.is_reject() && array.null_count() > 0 {
        return Err(SKError::NullValueInColumn {
            column: name.to_string(),
            policy: policy.name(),
        });
    }
    Ok(SKColumnData::Boolean(array.iter().collect()))
}
