//! Input validation shared by fit and transform
//! (spec `nearest-neighbor-imputation`).

use ndarray::ArrayView2;
use sciencekit_common::{SKError, SKFloat};

use crate::missingness::sk_is_missing;

/// Reject an empty dense input before any element is processed.
pub(super) fn sk_validated_shape<F>(
    features: &ArrayView2<'_, F>,
    operation: &'static str,
) -> Result<(usize, usize), SKError> {
    let (rows_count, columns_count) = features.dim();
    if rows_count == 0 || columns_count == 0 {
        return Err(SKError::EmptyInput { operation });
    }
    Ok((rows_count, columns_count))
}

/// Reject a non-finite value that the configured marker does not claim.
///
/// Under the `NaN` marker every `NaN` is missing by definition, so only
/// infinities are rejected. Under a scalar marker a `NaN` is a non-finite
/// value, so it is rejected too.
pub(super) fn sk_validate_finite<F: SKFloat>(
    features: &ArrayView2<'_, F>,
    marker: F,
) -> Result<(), SKError> {
    let columns_count = features.ncols();
    features
        .indexed_iter()
        .find(|(_, value)| !value.is_finite() && !sk_is_missing(**value, marker))
        .map(|((row, column), _)| SKError::NonFiniteValue {
            position: row * columns_count + column,
        })
        .map_or(Ok(()), Err)
}

/// Reject a donor request larger than the training pool.
///
/// A column may observe fewer rows than `nearest_neighbors_count`; those
/// columns simply contribute every donor they have. Only a count larger than
/// the whole training design is meaningless.
pub(super) fn sk_validate_neighbor_count(
    nearest_neighbors_count: usize,
    training_rows: usize,
) -> Result<(), SKError> {
    if nearest_neighbors_count > training_rows {
        return Err(SKError::InvalidHyperparameter {
            name: "nearest_neighbors_count",
            reason: format!(
                "{nearest_neighbors_count} donors requested but only {training_rows} training rows are available"
            ),
        });
    }
    Ok(())
}
