//! Input and hyperparameter validation for `SKTruncatedSVD` (spec
//! `truncated-svd`).

use sciencekit_common::{SKDataView, SKError, SKFloat, sk_validate_hyperparameter};

use super::config::{SKTruncatedSVDAlgorithm, SKTruncatedSVDConfiguration};

/// Returns `(number_of_rows, number_of_columns)` for a supported feature view.
pub(super) fn sk_view_dimensions<F>(
    features: &SKDataView<'_, F>,
) -> Result<(usize, usize), SKError> {
    match features {
        SKDataView::Dense(dense) => Ok(dense.dim()),
        SKDataView::Sparse(sparse) => Ok((sparse.rows(), sparse.cols())),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: features.representation(),
            suggestion: "convert the input to a supported dense or CSR view",
        }),
    }
}

/// Validates data-independent hyperparameters on `SKTruncatedSVDConfiguration`.
pub(super) fn sk_validate_configuration<F: SKFloat>(
    number_of_components: usize,
    algorithm: SKTruncatedSVDAlgorithm,
    number_of_iterations: usize,
    tolerance: F,
) -> Result<(), SKError> {
    sk_validate_hyperparameter(
        "number_of_components",
        number_of_components > 0,
        "must be strictly positive",
    )?;
    sk_validate_hyperparameter(
        "algorithm",
        matches!(algorithm, SKTruncatedSVDAlgorithm::Randomized),
        "only SKTruncatedSVDAlgorithm::Randomized is supported; ARPACK is deferred",
    )?;
    sk_validate_hyperparameter(
        "number_of_iterations",
        number_of_iterations > 0,
        "must be strictly positive",
    )?;
    sk_validate_hyperparameter(
        "tolerance",
        tolerance.is_finite() && tolerance >= F::zero(),
        "must be finite and non-negative",
    )
}

/// Validates shape, hyperparameters and finiteness before running `fit`.
pub(super) fn sk_validate_fit_input<F: SKFloat>(
    features: SKDataView<'_, F>,
    configuration: &SKTruncatedSVDConfiguration<F>,
    operation: &'static str,
) -> Result<(usize, usize), SKError> {
    sk_validate_configuration(
        configuration.number_of_components,
        configuration.algorithm,
        configuration.number_of_iterations,
        configuration.tolerance,
    )?;
    let (rows_count, columns_count) = sk_view_dimensions(&features)?;
    if rows_count == 0 || columns_count == 0 {
        return Err(SKError::EmptyInput { operation });
    }
    let maximum_components = rows_count.min(columns_count);
    if configuration.number_of_components > maximum_components {
        return Err(SKError::InvalidHyperparameter {
            name: "number_of_components",
            reason: format!(
                "requested {} components, which must lie in 1..={maximum_components} for shape ({rows_count}, {columns_count})",
                configuration.number_of_components
            ),
        });
    }
    sk_validate_finiteness(features)?;
    Ok((rows_count, columns_count))
}

/// Validates `transform` or `inverse_transform` input shape and finiteness.
#[allow(dead_code)]
pub(super) fn sk_validate_matrix_input<F: SKFloat>(
    features: SKDataView<'_, F>,
    expected_columns: usize,
    operation: &'static str,
) -> Result<(usize, usize), SKError> {
    let (rows_count, found_columns) = sk_view_dimensions(&features)?;
    if rows_count == 0 || found_columns == 0 {
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
/// position without densifying sparse inputs.
pub(super) fn sk_validate_finiteness<F: SKFloat>(
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
