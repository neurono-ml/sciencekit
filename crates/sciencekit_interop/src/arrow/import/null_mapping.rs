//! Mapping of Arrow validity bitmaps onto the declared missing-value policy
//! (spec `arrow-columnar-interop`).

use sciencekit_common::SKError;

use crate::missing::SKMissingValuePolicy;

/// Validate that a `ConstantFill` policy carries a finite fill value, guarding
/// against callers who construct the enum variant directly instead of going
/// through [`SKMissingValuePolicy::constant_fill`].
pub(super) fn sk_validate_constant_fill_policy(
    column_name: &str,
    policy: SKMissingValuePolicy,
) -> Result<(), SKError> {
    if let SKMissingValuePolicy::ConstantFill(value) = policy {
        if !value.is_finite() {
            return Err(SKError::InvalidHyperparameter {
                name: "constant_fill_value",
                reason: format!("the fill value {value} must be finite for column `{column_name}`"),
            });
        }
    }
    Ok(())
}

/// The policy value written into a floating column at a null position.
///
/// [`SKMissingValuePolicy::CategoricalMissingCode`] does not apply to numeric
/// columns; the null is retained so the numeric conversion rejects it later
/// with the policy name, keeping the decision in one place.
pub(super) fn sk_float_for_null(
    column_name: &str,
    policy: SKMissingValuePolicy,
) -> Result<Option<f64>, SKError> {
    sk_validate_constant_fill_policy(column_name, policy)?;
    match policy {
        SKMissingValuePolicy::Reject => Err(SKError::NullValueInColumn {
            column: column_name.to_string(),
            policy: policy.name(),
        }),
        SKMissingValuePolicy::NumericNaN => Ok(Some(f64::NAN)),
        SKMissingValuePolicy::ConstantFill(value) => Ok(Some(value)),
        SKMissingValuePolicy::CategoricalMissingCode => Ok(None),
    }
}

/// Narrow a finite policy fill value to `f32` without overflowing to infinity.
pub(super) fn sk_float32_from_fill(column_name: &str, value: f64) -> Result<f32, SKError> {
    let narrowed = value as f32;
    if !narrowed.is_finite() {
        return Err(SKError::InvalidHyperparameter {
            name: "constant_fill_value",
            reason: format!(
                "the fill value {value} is not representable as a finite f32 for column `{column_name}`"
            ),
        });
    }
    Ok(narrowed)
}
