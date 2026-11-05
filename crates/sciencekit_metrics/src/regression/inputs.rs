//! Shared input validation for the regression metric functions.

use ndarray::ArrayView1;
use sciencekit_common::{SKError, SKFloat};

use crate::validation::sk_validate_finite_values;

/// Validate equal, non-empty and finite target/prediction sequences.
pub(super) fn sk_validate_regression_inputs<F: SKFloat>(
    targets: &ArrayView1<F>,
    predictions: &ArrayView1<F>,
    operation: &'static str,
) -> Result<(), SKError> {
    if targets.is_empty() {
        return Err(SKError::EmptyInput { operation });
    }
    if targets.len() != predictions.len() {
        return Err(SKError::shape_mismatch_2d(
            targets.len(),
            1,
            predictions.len(),
            1,
        ));
    }
    sk_validate_finite_values(targets)?;
    sk_validate_finite_values(predictions)?;
    Ok(())
}
