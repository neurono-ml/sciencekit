//! Shared query validation (spec `nearest-neighbor-search`).

use ndarray::ArrayView2;
use sciencekit_common::{SKDataView, SKError, SKFloat};

/// Reject a sparse view before any distance is computed, pointing the caller at
/// the brute-force searcher that will serve sparse input.
pub(crate) fn sk_require_dense<F>(view: SKDataView<'_, F>) -> Result<ArrayView2<'_, F>, SKError> {
    match view {
        SKDataView::Dense(dense) => Ok(dense),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "csr",
            suggestion: "use the brute-force searcher for sparse input, delivered by the knn-classifier-regressor change",
        }),
    }
}

/// Validate a dense query batch against the index configuration.
///
/// Checks, in order: feature-count match, query finiteness, `neighbors_count`
/// strictly positive and not larger than the training-row count. Runs before
/// any distance is computed.
pub(crate) fn sk_validate_query<F: SKFloat>(
    queries: ArrayView2<'_, F>,
    training_features: usize,
    training_rows: usize,
    neighbors_count: usize,
) -> Result<(), SKError> {
    validate_shape_and_finiteness(queries, training_features)?;

    if neighbors_count == 0 {
        return Err(SKError::InvalidHyperparameter {
            name: "neighbors_count",
            reason: "must be strictly positive".to_string(),
        });
    }

    if neighbors_count > training_rows {
        return Err(SKError::InvalidHyperparameter {
            name: "neighbors_count",
            reason: format!(
                "requested {neighbors_count} neighbors from an index holding {training_rows} training rows"
            ),
        });
    }

    Ok(())
}

/// Validate a dense radius query batch and the radius itself.
pub(crate) fn sk_validate_radius<F: SKFloat>(
    queries: ArrayView2<'_, F>,
    training_features: usize,
    radius: F,
) -> Result<(), SKError> {
    validate_shape_and_finiteness(queries, training_features)?;

    if !radius.is_finite() || radius < F::zero() {
        return Err(SKError::InvalidHyperparameter {
            name: "radius",
            reason: "must be finite and non-negative".to_string(),
        });
    }

    Ok(())
}

fn validate_shape_and_finiteness<F: SKFloat>(
    queries: ArrayView2<'_, F>,
    training_features: usize,
) -> Result<(), SKError> {
    if queries.ncols() != training_features {
        return Err(SKError::ShapeMismatch {
            expected: vec![queries.nrows(), training_features],
            found: vec![queries.nrows(), queries.ncols()],
        });
    }

    for (position, value) in queries.iter().enumerate() {
        if !value.is_finite() {
            return Err(SKError::NonFiniteValue { position });
        }
    }

    Ok(())
}
