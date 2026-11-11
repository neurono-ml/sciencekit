//! Input validation for MiniBatchKMeans fit, `partial_fit` and query paths
//! (spec `mini-batch-k-means`, decision 8).

use sciencekit_common::{SKDataView, SKError, SKFloat};

use crate::kmeans::distance::sk_view_dimensions;

/// Validates an initial training or mini-batch view against the cluster count.
pub(crate) fn sk_validate_initial_batch<F: SKFloat>(
    features: SKDataView<'_, F>,
    number_of_clusters: usize,
    operation: &'static str,
) -> Result<(), SKError> {
    sk_validate_supported_representation(&features)?;
    let (number_of_rows, number_of_features) = sk_view_dimensions(&features);
    if number_of_rows == 0 || number_of_features == 0 {
        return Err(SKError::EmptyInput { operation });
    }
    if number_of_rows < number_of_clusters {
        return Err(SKError::InvalidHyperparameter {
            name: "number_of_clusters",
            reason: format!(
                "requested {number_of_clusters} clusters from {number_of_rows} batch rows"
            ),
        });
    }
    sk_validate_finiteness(features)
}

/// Validates a continuation `partial_fit` or streaming batch against the
/// fitted feature width and cluster count.
pub(crate) fn sk_validate_continuation_batch<F: SKFloat>(
    features: SKDataView<'_, F>,
    expected_features: usize,
    number_of_clusters: usize,
    enforce_cluster_floor: bool,
    operation: &'static str,
) -> Result<(), SKError> {
    sk_validate_supported_representation(&features)?;
    let (number_of_rows, found_features) = sk_view_dimensions(&features);
    if number_of_rows == 0 || found_features == 0 {
        return Err(SKError::EmptyInput { operation });
    }
    if found_features != expected_features {
        return Err(SKError::ShapeMismatch {
            expected: vec![number_of_rows, expected_features],
            found: vec![number_of_rows, found_features],
        });
    }
    if enforce_cluster_floor && number_of_rows < number_of_clusters {
        return Err(SKError::InvalidHyperparameter {
            name: "number_of_clusters",
            reason: format!(
                "batch has {number_of_rows} rows, fewer than number_of_clusters ({number_of_clusters})"
            ),
        });
    }
    sk_validate_finiteness(features)
}

/// Rejects representations outside Dense and CSR.
pub(crate) fn sk_validate_supported_representation<F>(
    features: &SKDataView<'_, F>,
) -> Result<(), SKError> {
    match features {
        SKDataView::Dense(_) | SKDataView::Sparse(_) => Ok(()),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: features.representation(),
            suggestion: "convert the input to a supported dense or CSR view",
        }),
    }
}

/// Rejects any NaN or infinite entry, reporting its flat row-major position.
pub(crate) fn sk_validate_finiteness<F: SKFloat>(
    features: SKDataView<'_, F>,
) -> Result<(), SKError> {
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
                if let Some((column, _)) = stored.iter().find(|(_, value)| !value.is_finite()) {
                    return Err(SKError::NonFiniteValue {
                        position: row * number_of_columns + column,
                    });
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
