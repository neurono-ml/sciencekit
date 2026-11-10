//! Shared query validation (spec `nearest-neighbor-search`).

use ndarray::ArrayView2;
use sciencekit_common::{SKError, SKFloat};

/// Validate a dense query batch against the index configuration.
///
/// Checks, in order: feature-count match, query finiteness, `neighbors_count`
/// strictly positive and not larger than the training-row count. Runs before
/// any distance is computed.
pub fn sk_validate_query<F: SKFloat>(
    queries: ArrayView2<'_, F>,
    training_features: usize,
    training_rows: usize,
    neighbors_count: usize,
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
