//! Single-dispatch input preparation and hyperparameter validation for
//! `SKTruncatedSVD` (spec `truncated-svd` and `data-view-boundary`).

use ndarray::ArrayView2;
use sciencekit_common::{SKDataView, SKError, SKFloat, sk_validate_hyperparameter};
use sprs::CsMatView;

use super::config::{SKTruncatedSVDAlgorithm, SKTruncatedSVDConfiguration};

/// Borrowed dense or compressed-row matrix operand extracted by dispatching on
/// [`SKDataView`] once at the start of each public operation.
#[derive(Clone, Copy)]
pub(super) enum SKTruncatedInput<'a, F: Copy> {
    Dense(ArrayView2<'a, F>),
    SparseCsr(CsMatView<'a, F>),
}

impl<'a, F: Copy> SKTruncatedInput<'a, F> {
    /// Dispatches on `SKDataView` once per operation, rejecting CSC or unknown
    /// variants up front before scanning any element.
    pub(super) fn try_from_view(features: SKDataView<'a, F>) -> Result<Self, SKError> {
        match features {
            SKDataView::Dense(dense) => Ok(Self::Dense(dense)),
            SKDataView::Sparse(sparse) => {
                if !sparse.is_csr() {
                    return Err(SKError::UnsupportedRepresentation {
                        representation: "csc",
                        suggestion: "convert the sparse matrix to compressed-row (CSR) format with to_csr()",
                    });
                }
                Ok(Self::SparseCsr(sparse))
            }
            _ => Err(SKError::UnsupportedRepresentation {
                representation: features.representation(),
                suggestion: "convert the input to a supported dense or CSR view",
            }),
        }
    }

    /// Logical `(number_of_rows, number_of_columns)` dimensions.
    pub(super) fn dimensions(&self) -> (usize, usize) {
        match self {
            Self::Dense(dense) => dense.dim(),
            Self::SparseCsr(sparse) => (sparse.rows(), sparse.cols()),
        }
    }

    /// Number of stored scalar elements (`rows * columns` for dense, `nnz` for CSR).
    pub(super) fn stored_elements(&self) -> u64 {
        match self {
            Self::Dense(dense) => {
                let (rows, columns) = dense.dim();
                (rows as u64).saturating_mul(columns as u64)
            }
            Self::SparseCsr(sparse) => sparse.nnz() as u64,
        }
    }

    /// Re-borrows as an [`SKDataView`] when delegating to `sciencekit_math`.
    pub(super) fn as_data_view(self) -> SKDataView<'a, F> {
        match self {
            Self::Dense(dense) => SKDataView::Dense(dense),
            Self::SparseCsr(sparse) => SKDataView::Sparse(sparse),
        }
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
    input: SKTruncatedInput<'_, F>,
    configuration: &SKTruncatedSVDConfiguration<F>,
    operation: &'static str,
) -> Result<(usize, usize), SKError> {
    sk_validate_configuration(
        configuration.number_of_components,
        configuration.algorithm,
        configuration.number_of_iterations,
        configuration.tolerance,
    )?;
    let (rows_count, columns_count) = input.dimensions();
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
    sk_validate_finiteness(input)?;
    Ok((rows_count, columns_count))
}

/// Validates `transform` or `inverse_transform` input shape and finiteness.
pub(super) fn sk_validate_matrix_input<F: SKFloat>(
    input: SKTruncatedInput<'_, F>,
    expected_columns: usize,
    operation: &'static str,
) -> Result<(usize, usize), SKError> {
    let (rows_count, found_columns) = input.dimensions();
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
    sk_validate_finiteness(input)?;
    Ok((rows_count, found_columns))
}

/// Rejects any non-finite value (`NaN` or `±∞`), reporting its flat row-major
/// position without densifying sparse inputs.
pub(super) fn sk_validate_finiteness<F: SKFloat>(
    input: SKTruncatedInput<'_, F>,
) -> Result<(), SKError> {
    let non_finite_position = match input {
        SKTruncatedInput::Dense(dense) => {
            let columns_count = dense.ncols();
            dense.indexed_iter().find_map(|((row, column), &value)| {
                (!value.is_finite()).then_some(row * columns_count + column)
            })
        }
        SKTruncatedInput::SparseCsr(sparse) => {
            let columns_count = sparse.cols();
            sparse
                .outer_iterator()
                .enumerate()
                .find_map(|(row, stored_row)| {
                    stored_row.iter().find_map(|(column, &value)| {
                        (!value.is_finite()).then_some(row * columns_count + column)
                    })
                })
        }
    };
    match non_finite_position {
        Some(position) => Err(SKError::NonFiniteValue { position }),
        None => Ok(()),
    }
}
