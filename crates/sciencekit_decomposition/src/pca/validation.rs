//! Input validation for PCA fitting, projection and inverse projection (spec
//! `principal-component-analysis`, decision 8).

use sciencekit_common::{SKDataView, SKError, SKFloat};

use super::centering::sk_view_dimensions;

/// Validates the shape and requested component count (`None` →
/// `min(n_samples, n_features)`) before any element or representation scan.
pub(super) fn sk_validate_fit_shape<F: SKFloat>(
    features: &SKDataView<'_, F>,
    requested_components: Option<usize>,
    operation: &'static str,
) -> Result<(usize, usize, usize), SKError> {
    let (rows_count, columns_count) = sk_view_dimensions(features)?;
    if rows_count == 0 || columns_count == 0 {
        return Err(SKError::EmptyInput { operation });
    }
    let maximum_components = rows_count.min(columns_count);
    let resolved_components = requested_components.unwrap_or(maximum_components);
    if resolved_components == 0 || resolved_components > maximum_components {
        return Err(SKError::InvalidHyperparameter {
            name: "number_of_components",
            reason: format!(
                "requested {resolved_components} components, which must lie in 1..={maximum_components} for shape ({rows_count}, {columns_count})"
            ),
        });
    }
    Ok((rows_count, columns_count, resolved_components))
}

/// Validates a `transform` or `inverse_transform` input against the expected
/// column count and checks finiteness.
pub(crate) fn sk_validate_matrix_input<F: SKFloat>(
    features: SKDataView<'_, F>,
    expected_columns: usize,
    operation: &'static str,
) -> Result<(usize, usize), SKError> {
    let (rows_count, found_columns) = sk_view_dimensions(&features)?;
    if rows_count == 0 {
        return Err(SKError::EmptyInput { operation });
    }
    if found_columns != expected_columns {
        return Err(SKError::shape_mismatch_2d(
            rows_count,
            expected_columns,
            rows_count,
            found_columns,
        ));
    }
    sk_validate_finiteness(features)?;
    Ok((rows_count, found_columns))
}

/// Rejects any non-finite value (`NaN` or `±∞`), reporting its flat row-major
/// position.
pub(crate) fn sk_validate_finiteness<F: SKFloat>(
    features: SKDataView<'_, F>,
) -> Result<(), SKError> {
    match features {
        SKDataView::Dense(dense) => {
            let columns_count = dense.ncols();
            for ((row, column), &value) in dense.indexed_iter() {
                if !value.is_finite() {
                    return Err(SKError::NonFiniteValue {
                        position: row * columns_count + column,
                    });
                }
            }
            Ok(())
        }
        SKDataView::Sparse(sparse) => {
            let columns_count = sparse.cols();
            for (row, stored_row) in sparse.outer_iterator().enumerate() {
                for (column, &value) in stored_row.iter() {
                    if !value.is_finite() {
                        return Err(SKError::NonFiniteValue {
                            position: row * columns_count + column,
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
