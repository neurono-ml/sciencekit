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

/// Convert a boolean column into `0`/`1` category codes.
///
/// Boolean features are numeric (`0`/`1`) by default; a caller that wants them
/// on the categorical path uses this conversion instead. Missing entries carry
/// the documented categorical missing code once the categorical extraction
/// applies [`SKMissingValuePolicy::CategoricalMissingCode`].
///
/// # Errors
/// Returns [`SKError::UnsupportedColumnType`] for a non-boolean column and
/// [`SKError::NullValueInColumn`] when a null meets a policy other than the
/// categorical missing code.
pub fn sk_boolean_category_codes(
    column_name: &str,
    column: &SKColumnData,
    policy: SKMissingValuePolicy,
) -> Result<SKColumnData, SKError> {
    let values = match column {
        SKColumnData::Boolean(values) => values,
        other => {
            return Err(SKError::UnsupportedColumnType {
                column: column_name.to_string(),
                logical_type: other.logical_type().to_string(),
                suggestion: "select a boolean column to code it as 0/1 categories".to_string(),
            });
        }
    };
    let codes = values
        .iter()
        .map(|value| match value {
            Some(true) => Ok(Some(1)),
            Some(false) => Ok(Some(0)),
            None => match policy {
                SKMissingValuePolicy::CategoricalMissingCode => Ok(None),
                _ => Err(SKError::NullValueInColumn {
                    column: column_name.to_string(),
                    policy: policy.name(),
                }),
            },
        })
        .collect::<Result<Vec<_>, SKError>>()?;
    Ok(SKColumnData::CategoryCode(codes))
}
