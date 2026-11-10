//! The documented missing-value policy (spec `dataset-interop-core`).

use std::fmt;

use sciencekit_common::SKError;

/// How a conversion treats null entries in a nullable column.
///
/// Every conversion applies exactly one declared policy. The default is
/// [`SKMissingValuePolicy::Reject`], so a caller must acknowledge nullability
/// explicitly rather than inheriting an engine-specific behavior.
#[derive(Debug, Clone, Copy, PartialEq)]
#[non_exhaustive]
pub enum SKMissingValuePolicy {
    /// Reject the conversion on the first null encountered.
    Reject,
    /// Map numeric nulls to the floating-point not-a-number sentinel.
    ///
    /// Integer columns carrying nulls are promoted to the model's floating
    /// scalar before the sentinel is written, never truncated.
    NumericNaN,
    /// Map null nominal/category-coded entries to the documented missing code
    /// [`sciencekit_common::SK_CATEGORICAL_MISSING_CODE`].
    CategoricalMissingCode,
    /// Replace numeric nulls with a caller-provided constant.
    ConstantFill(f64),
}

impl Default for SKMissingValuePolicy {
    fn default() -> Self {
        SKMissingValuePolicy::Reject
    }
}

impl SKMissingValuePolicy {
    /// Build a constant-fill policy from a numeric replacement value.
    ///
    /// # Errors
    /// Returns [`SKError::InvalidHyperparameter`] when the value is not finite:
    /// a non-finite `NaN`/infinite constant would silently mimic the missing
    /// sentinel or overflow the model scalar.
    pub fn constant_fill(value: f64) -> Result<Self, SKError> {
        if value.is_finite() {
            Ok(SKMissingValuePolicy::ConstantFill(value))
        } else {
            Err(SKError::InvalidHyperparameter {
                name: "constant_fill_value",
                reason: "the fill value must be finite".to_string(),
            })
        }
    }

    /// The stable, lower-case policy name used in errors and traces.
    pub fn name(&self) -> &'static str {
        match self {
            SKMissingValuePolicy::Reject => "reject",
            SKMissingValuePolicy::NumericNaN => "numeric_nan",
            SKMissingValuePolicy::CategoricalMissingCode => "categorical_missing_code",
            SKMissingValuePolicy::ConstantFill(_) => "constant_fill",
        }
    }

    /// Whether the policy rejects every null.
    pub fn is_reject(&self) -> bool {
        matches!(self, SKMissingValuePolicy::Reject)
    }

    /// The constant fill value, when the policy is constant fill.
    pub fn fill_value(&self) -> Option<f64> {
        match self {
            SKMissingValuePolicy::ConstantFill(value) => Some(*value),
            _ => None,
        }
    }
}

impl fmt::Display for SKMissingValuePolicy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.name())
    }
}
