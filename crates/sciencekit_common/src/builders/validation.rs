//! Shared hyperparameter validation (spec `base-builders`).

use crate::SKError;

/// Validate a named hyperparameter, failing with
/// [`SKError::InvalidHyperparameter`] when the condition is false.
///
/// Shared by every builder so invalid configuration surfaces as an error at
/// `build()` time instead of panicking.
pub fn sk_validate_hyperparameter(
    name: &'static str,
    valid: bool,
    reason: impl Into<String>,
) -> Result<(), SKError> {
    if valid {
        Ok(())
    } else {
        Err(SKError::InvalidHyperparameter {
            name,
            reason: reason.into(),
        })
    }
}
