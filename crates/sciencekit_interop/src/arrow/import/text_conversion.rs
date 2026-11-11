//! Import of Arrow text columns into the nominal logical type (spec
//! `arrow-columnar-interop`).

use arrow_array::{Array, LargeStringArray, StringArray, StringViewArray};
use sciencekit_common::SKError;

use crate::columns::SKColumnData;
use crate::missing::SKMissingValuePolicy;

fn sk_check_nulls(
    name: &str,
    array: &dyn Array,
    policy: SKMissingValuePolicy,
) -> Result<(), SKError> {
    if policy.is_reject() && array.null_count() > 0 {
        return Err(SKError::NullValueInColumn {
            column: name.to_string(),
            policy: policy.name(),
        });
    }
    Ok(())
}

/// Convert a `Utf8` Arrow column into owned nominal text.
///
/// # Errors
/// Returns [`SKError::NullValueInColumn`] under the reject policy.
pub(super) fn sk_string_column(
    name: &str,
    array: &StringArray,
    policy: SKMissingValuePolicy,
) -> Result<SKColumnData, SKError> {
    sk_check_nulls(name, array, policy)?;
    Ok(SKColumnData::Nominal(
        array.iter().map(|value| value.map(str::to_owned)).collect(),
    ))
}

/// Convert a `LargeUtf8` Arrow column into owned nominal text.
///
/// # Errors
/// Returns [`SKError::NullValueInColumn`] under the reject policy.
pub(super) fn sk_large_string_column(
    name: &str,
    array: &LargeStringArray,
    policy: SKMissingValuePolicy,
) -> Result<SKColumnData, SKError> {
    sk_check_nulls(name, array, policy)?;
    Ok(SKColumnData::Nominal(
        array.iter().map(|value| value.map(str::to_owned)).collect(),
    ))
}

/// Convert a `Utf8View` Arrow column into owned nominal text.
///
/// # Errors
/// Returns [`SKError::NullValueInColumn`] under the reject policy.
pub(super) fn sk_string_view_column(
    name: &str,
    array: &StringViewArray,
    policy: SKMissingValuePolicy,
) -> Result<SKColumnData, SKError> {
    sk_check_nulls(name, array, policy)?;
    Ok(SKColumnData::Nominal(
        array.iter().map(|value| value.map(str::to_owned)).collect(),
    ))
}
