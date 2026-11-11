//! Single-dispatch dense/CSR multiplication and non-densifying variance
//! computation for `SKTruncatedSVD` (spec `truncated-svd`, decisions 2 and 3).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, azip};
use rayon::prelude::*;
use sciencekit_common::{SKBackendKind, SKDataView, SKError, SKFloat};
use sciencekit_math::backend::{SKBackendSelection, sk_math_backend_for_kind};
use sciencekit_math::sparse_ops::sk_csr_dense_product;

const ROW_BLOCK_SIZE: usize = 256;

/// Computes `left · right_factor` with a single representation dispatch per
/// call, never materializing a dense copy of a sparse CSR input.
pub(super) fn sk_multiply_view_by_dense<F: SKBackendSelection>(
    left: SKDataView<'_, F>,
    right_factor: ArrayView2<'_, F>,
    backend_kind: SKBackendKind,
    parallelism: usize,
) -> Result<Array2<F>, SKError> {
    match left {
        SKDataView::Dense(dense) => {
            let rows_count = dense.nrows();
            let backend = sk_math_backend_for_kind::<F>(backend_kind);
            let mut output = Array2::<F>::zeros((rows_count, right_factor.ncols()));
            if parallelism > 1 && rows_count > ROW_BLOCK_SIZE {
                dense
                    .axis_chunks_iter(Axis(0), ROW_BLOCK_SIZE)
                    .into_par_iter()
                    .zip(
                        output
                            .axis_chunks_iter_mut(Axis(0), ROW_BLOCK_SIZE)
                            .into_par_iter(),
                    )
                    .for_each(|(input_chunk, mut output_chunk)| {
                        let block = backend.gemm(input_chunk, right_factor, F::one(), 1);
                        output_chunk.assign(&block);
                    });
            } else {
                for (input_chunk, mut output_chunk) in dense
                    .axis_chunks_iter(Axis(0), ROW_BLOCK_SIZE)
                    .zip(output.axis_chunks_iter_mut(Axis(0), ROW_BLOCK_SIZE))
                {
                    let block = backend.gemm(input_chunk, right_factor, F::one(), 1);
                    output_chunk.assign(&block);
                }
            }
            Ok(output)
        }
        SKDataView::Sparse(sparse) => Ok(sk_csr_dense_product(&sparse, &right_factor)),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: left.representation(),
            suggestion: "convert the input to a supported dense or CSR view",
        }),
    }
}

/// Computes `explained_variance` (population variance of each column of
/// `projected`) and `explained_variance_ratio` (divided by the sum of the input
/// column variances), without densifying sparse inputs.
pub(super) fn sk_variance_and_ratio<F: SKFloat>(
    projected: ArrayView2<'_, F>,
    features: SKDataView<'_, F>,
    parallelism: usize,
) -> Result<(Array1<F>, Array1<F>), SKError> {
    let explained_variance = sk_dense_column_variances(projected, parallelism);
    let total_variance = sk_total_input_variance(features, parallelism)?;
    if total_variance <= F::zero() {
        let zeros = Array1::<F>::zeros(explained_variance.len());
        return Ok((explained_variance, zeros));
    }
    let mut ratios = explained_variance.mapv(|variance| variance / total_variance);
    let ratio_sum = ratios
        .iter()
        .fold(F::zero(), |accumulator, &value| accumulator + value);
    if ratio_sum > F::one() {
        let excess = ratio_sum - F::one();
        if let Some(first) = ratios.first_mut() {
            *first = (*first - excess).max(F::zero());
        }
    }
    Ok((explained_variance, ratios))
}

fn sk_dense_column_variances<F: SKFloat>(
    matrix: ArrayView2<'_, F>,
    parallelism: usize,
) -> Array1<F> {
    let (rows_count, columns_count) = matrix.dim();
    if rows_count <= 1 || columns_count == 0 {
        return Array1::<F>::zeros(columns_count);
    }
    let rows_scalar = F::from(rows_count).expect("row count fits in scalar");
    let sum_partials = collect_chunk_partials(matrix, parallelism, sk_sum_chunk_columns);
    let means = reduce_partials(sum_partials, columns_count).mapv(|sum| sum / rows_scalar);
    let mean_view = means.view();
    let squared_partials = collect_chunk_partials(matrix, parallelism, |chunk| {
        sk_sum_chunk_squared_deviations(chunk, mean_view)
    });
    reduce_partials(squared_partials, columns_count).mapv(|sum| sum / rows_scalar)
}

fn collect_chunk_partials<F, G>(
    matrix: ArrayView2<'_, F>,
    parallelism: usize,
    reducer: G,
) -> Vec<Array1<F>>
where
    F: SKFloat,
    G: Fn(ArrayView2<'_, F>) -> Array1<F> + Sync + Send,
{
    if parallelism > 1 && matrix.nrows() > ROW_BLOCK_SIZE {
        matrix
            .axis_chunks_iter(Axis(0), ROW_BLOCK_SIZE)
            .into_par_iter()
            .map(reducer)
            .collect()
    } else {
        matrix
            .axis_chunks_iter(Axis(0), ROW_BLOCK_SIZE)
            .map(reducer)
            .collect()
    }
}

fn reduce_partials<F: SKFloat>(partials: Vec<Array1<F>>, columns_count: usize) -> Array1<F> {
    let mut total = Array1::<F>::zeros(columns_count);
    for partial in partials {
        azip!((accumulator in &mut total, &value in &partial) {
            *accumulator = *accumulator + value;
        });
    }
    total
}

fn sk_sum_chunk_columns<F: SKFloat>(chunk: ArrayView2<'_, F>) -> Array1<F> {
    let mut partial = Array1::<F>::zeros(chunk.ncols());
    for row in chunk.rows() {
        azip!((accumulator in &mut partial, &value in row) {
            *accumulator = *accumulator + value;
        });
    }
    partial
}

fn sk_sum_chunk_squared_deviations<F: SKFloat>(
    chunk: ArrayView2<'_, F>,
    means: ArrayView1<'_, F>,
) -> Array1<F> {
    let mut partial = Array1::<F>::zeros(chunk.ncols());
    for row in chunk.rows() {
        azip!((accumulator in &mut partial, &value in row, &mean in means) {
            let deviation = value - mean;
            *accumulator = *accumulator + deviation * deviation;
        });
    }
    partial
}

fn sk_total_input_variance<F: SKFloat>(
    features: SKDataView<'_, F>,
    parallelism: usize,
) -> Result<F, SKError> {
    match features {
        SKDataView::Dense(dense) => Ok(sk_dense_column_variances(dense, parallelism)
            .iter()
            .fold(F::zero(), |accumulator, &value| accumulator + value)),
        SKDataView::Sparse(sparse) => {
            let (rows_count, columns_count) = (sparse.rows(), sparse.cols());
            if rows_count <= 1 || columns_count == 0 {
                return Ok(F::zero());
            }
            let rows_scalar = F::from(rows_count).expect("row count fits in scalar");
            let mut sums = Array1::<F>::zeros(columns_count);
            for row_view in sparse.outer_iterator() {
                for (column, &value) in row_view.iter() {
                    sums[column] = sums[column] + value;
                }
            }
            let means = sums.mapv(|sum| sum / rows_scalar);
            let mut stored_squares = vec![F::zero(); columns_count];
            let mut nonzero_counts = vec![0usize; columns_count];
            for row_view in sparse.outer_iterator() {
                for (column, &value) in row_view.iter() {
                    let deviation = value - means[column];
                    stored_squares[column] = stored_squares[column] + deviation * deviation;
                    nonzero_counts[column] += 1;
                }
            }
            let sum_of_squares = stored_squares
                .into_iter()
                .zip(nonzero_counts)
                .zip(means.iter())
                .fold(F::zero(), |accumulator, ((stored, count), &mean)| {
                    let zero_rows = F::from(rows_count - count).expect("row count fits in scalar");
                    accumulator + stored + zero_rows * mean * mean
                });
            Ok(sum_of_squares / rows_scalar)
        }
        _ => Err(SKError::UnsupportedRepresentation {
            representation: features.representation(),
            suggestion: "convert the input to a supported dense or CSR view",
        }),
    }
}
