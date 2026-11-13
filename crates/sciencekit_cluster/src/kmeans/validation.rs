//! Fit and assignment validation (spec `k-means`, decision 9).

use sciencekit_common::{SKDataView, SKError, SKFloat};

use super::distance::sk_view_dimensions;

/// Validates a training input and the cluster count before any distance is
/// computed.
pub(crate) fn sk_validate_fit_features<F: SKFloat>(
    features: SKDataView<'_, F>,
    number_of_clusters: usize,
) -> Result<(), SKError> {
    let (number_of_rows, number_of_features) = sk_view_dimensions(&features);
    if number_of_rows == 0 || number_of_features == 0 {
        return Err(SKError::EmptyInput { operation: "fit" });
    }
    if number_of_clusters > number_of_rows {
        return Err(SKError::InvalidHyperparameter {
            name: "number_of_clusters",
            reason: format!(
                "requested {number_of_clusters} clusters from {number_of_rows} training rows"
            ),
        });
    }
    sk_validate_finiteness(features)
}

/// Validates an assignment input against the fitted feature count.
pub(crate) fn sk_validate_query_features<F: SKFloat>(
    features: SKDataView<'_, F>,
    number_of_features: usize,
) -> Result<(), SKError> {
    let (number_of_rows, found_features) = sk_view_dimensions(&features);
    if found_features != number_of_features {
        return Err(SKError::ShapeMismatch {
            expected: vec![number_of_rows, number_of_features],
            found: vec![number_of_rows, found_features],
        });
    }
    sk_validate_finiteness(features)
}

/// Rejects a non-finite value, naming its flat dense position.
fn sk_validate_finiteness<F: SKFloat>(features: SKDataView<'_, F>) -> Result<(), SKError> {
    match features {
        SKDataView::Dense(dense) => {
            if let Some(position) = dense.iter().position(|value| !value.is_finite()) {
                return Err(SKError::NonFiniteValue { position });
            }
            Ok(())
        }
        SKDataView::Sparse(sparse) => {
            let number_of_columns = sparse.cols();
            for (row, stored) in sparse.outer_iterator().enumerate() {
                for (column, value) in stored.iter() {
                    if !value.is_finite() {
                        return Err(SKError::NonFiniteValue {
                            position: row * number_of_columns + column,
                        });
                    }
                }
            }
            Ok(())
        }
        _ => Err(SKError::UnsupportedRepresentation {
            representation: features.representation(),
            suggestion: "convert the input to a supported dense or CSR view",
        }),
    }
}
