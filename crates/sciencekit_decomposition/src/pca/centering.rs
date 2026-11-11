//! Feature-mean computation, centering, total centered variance and the
//! explicit sparse-densification gate (spec `principal-component-analysis`,
//! decision 2).

use std::mem::size_of;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, azip};
use rayon::prelude::*;
use sciencekit_common::{SKDataView, SKError, SKFloat};

const MEAN_REDUCTION_ROW_CHUNK: usize = 256;

/// Suggestion string returned when a dense-only solver rejects a sparse view.
pub(super) const SPARSE_SOLVER_SUGGESTION: &str = "use SKPCASolver::Randomized for implicit sparse centering, SKTruncatedSVD for uncentered sparse decomposition, or set allow_densification(true)";

/// Returns `(number_of_rows, number_of_columns)` for a supported feature view.
pub(crate) fn sk_view_dimensions<F>(
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

/// Computes the per-feature arithmetic mean across all rows without densifying
/// a sparse matrix, using fixed-size row blocks on dense matrices so parallel
/// and sequential reductions yield bit-identical sums.
pub(super) fn sk_feature_means<F: SKFloat>(
    features: SKDataView<'_, F>,
    parallelism: usize,
) -> Result<Array1<F>, SKError> {
    let (rows_count, columns_count) = sk_view_dimensions(&features)?;
    if rows_count == 0 || columns_count == 0 {
        return Err(SKError::EmptyInput { operation: "fit" });
    }
    let rows_scalar = F::from(rows_count).expect("row count fits in scalar");
    match features {
        SKDataView::Dense(dense) => {
            let partials: Vec<Array1<F>> =
                if parallelism > 1 && rows_count > MEAN_REDUCTION_ROW_CHUNK {
                    dense
                        .axis_chunks_iter(Axis(0), MEAN_REDUCTION_ROW_CHUNK)
                        .into_par_iter()
                        .map(sk_sum_dense_chunk)
                        .collect()
                } else {
                    dense
                        .axis_chunks_iter(Axis(0), MEAN_REDUCTION_ROW_CHUNK)
                        .map(sk_sum_dense_chunk)
                        .collect()
                };
            let mut sums = Array1::<F>::zeros(columns_count);
            for partial in partials {
                azip!((accumulator in &mut sums, &value in &partial) {
                    *accumulator = *accumulator + value;
                });
            }
            Ok(sums.mapv(|sum| sum / rows_scalar))
        }
        SKDataView::Sparse(sparse) => {
            let mut sums = Array1::<F>::zeros(columns_count);
            for row_view in sparse.outer_iterator() {
                for (column, &value) in row_view.iter() {
                    sums[column] = sums[column] + value;
                }
            }
            Ok(sums.mapv(|sum| sum / rows_scalar))
        }
        _ => Err(SKError::UnsupportedRepresentation {
            representation: features.representation(),
            suggestion: "convert the input to a supported dense or CSR view",
        }),
    }
}

fn sk_sum_dense_chunk<F: SKFloat>(chunk: ArrayView2<'_, F>) -> Array1<F> {
    let mut partial = Array1::<F>::zeros(chunk.ncols());
    for row in chunk.rows() {
        azip!((accumulator in &mut partial, &value in row) {
            *accumulator = *accumulator + value;
        });
    }
    partial
}

/// Materializes `X - 1 μᵀ` from a dense feature view.
pub(crate) fn sk_center_dense<F: SKFloat>(
    dense: ArrayView2<'_, F>,
    mean: ArrayView1<'_, F>,
) -> Array2<F> {
    let mut centered = dense.to_owned();
    for mut row in centered.rows_mut() {
        azip!((value in &mut row, column_mean in mean) {
            *value = *value - *column_mean;
        });
    }
    centered
}

/// Computes the total unbiased variance `‖X − 1 μᵀ‖_F² / (n − 1)` across all
/// features without materializing a dense matrix for sparse inputs.
pub(super) fn sk_total_centered_variance<F: SKFloat>(
    features: SKDataView<'_, F>,
    mean: ArrayView1<'_, F>,
) -> Result<F, SKError> {
    let (rows_count, columns_count) = sk_view_dimensions(&features)?;
    if rows_count <= 1 {
        return Ok(F::zero());
    }
    let degrees_of_freedom = F::from(rows_count - 1).expect("degrees of freedom fit in scalar");
    let sum_of_squares = match features {
        SKDataView::Dense(dense) => dense.rows().into_iter().fold(F::zero(), |total, row| {
            let mut row_sum = F::zero();
            azip!((&value in row, &column_mean in mean) {
                let deviation = value - column_mean;
                row_sum = row_sum + deviation * deviation;
            });
            total + row_sum
        }),
        SKDataView::Sparse(sparse) => {
            let mut column_sum_squares = vec![F::zero(); columns_count];
            let mut nonzero_counts = vec![0usize; columns_count];
            for row_view in sparse.outer_iterator() {
                for (column, &value) in row_view.iter() {
                    let deviation = value - mean[column];
                    column_sum_squares[column] = column_sum_squares[column] + deviation * deviation;
                    nonzero_counts[column] += 1;
                }
            }
            column_sum_squares
                .into_iter()
                .zip(nonzero_counts)
                .zip(mean.iter())
                .fold(F::zero(), |accumulator, ((stored, count), &column_mean)| {
                    let zero_rows = F::from(rows_count - count).expect("row count fits in scalar");
                    accumulator + stored + zero_rows * column_mean * column_mean
                })
        }
        _ => {
            return Err(SKError::UnsupportedRepresentation {
                representation: features.representation(),
                suggestion: "convert the input to a supported dense or CSR view",
            });
        }
    };
    Ok(sum_of_squares / degrees_of_freedom)
}

/// Returns the centered dense matrix `X − 1 μᵀ`, rejecting sparse inputs unless
/// `allow_densification` is explicitly enabled (emitting a byte-cost warning).
pub(super) fn sk_centered_dense_or_reject<F: SKFloat>(
    features: SKDataView<'_, F>,
    mean: ArrayView1<'_, F>,
    allow_densification: bool,
) -> Result<Array2<F>, SKError> {
    match features {
        SKDataView::Dense(dense) => Ok(sk_center_dense(dense, mean)),
        SKDataView::Sparse(_) if !allow_densification => Err(SKError::UnsupportedRepresentation {
            representation: "csr",
            suggestion: SPARSE_SOLVER_SUGGESTION,
        }),
        SKDataView::Sparse(sparse) => {
            let (rows, columns) = (sparse.rows(), sparse.cols());
            let estimated_dense_bytes = (rows as u64)
                .saturating_mul(columns as u64)
                .saturating_mul(size_of::<F>() as u64);
            tracing::warn!(
                rows = rows,
                columns = columns,
                estimated_dense_bytes = estimated_dense_bytes,
                "densifying sparse CSR input for dense PCA solver because allow_densification(true) is enabled"
            );
            let mut centered = Array2::<F>::zeros((rows, columns));
            for (mut target_row, sparse_row) in
                centered.rows_mut().into_iter().zip(sparse.outer_iterator())
            {
                azip!((cell in &mut target_row, &column_mean in mean) {
                    *cell = -column_mean;
                });
                for (column, &value) in sparse_row.iter() {
                    target_row[column] = target_row[column] + value;
                }
            }
            Ok(centered)
        }
        _ => Err(SKError::UnsupportedRepresentation {
            representation: features.representation(),
            suggestion: "convert the input to a supported dense or CSR view",
        }),
    }
}
