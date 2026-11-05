//! Shared numeric validation for the metric functions.

use ndarray::ArrayView1;
use sciencekit_common::{SKError, SKFloat};

/// Reject a non-finite value, reporting its zero-based position.
pub(crate) fn sk_validate_finite_values<F: SKFloat>(values: &ArrayView1<F>) -> Result<(), SKError> {
    for (position, value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(SKError::NonFiniteValue { position });
        }
    }
    Ok(())
}
