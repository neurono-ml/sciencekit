//! Chan-Golub-LeVeque scatter-matrix merging and dense batch moment reduction
//! (spec `incremental-pca`, decisions 1 and 2).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, azip};
use rayon::prelude::*;
use sciencekit_common::SKFloat;
use sciencekit_math::backend::SKMathBackend;

use super::accumulator::SKCovarianceAccumulator;

const DENSE_MOMENT_ROW_CHUNK: usize = 256;

/// Merges a precomputed batch `(batch_count, batch_mean, batch_scatter)` into
/// the running `(samples_seen, mean, scatter)` state without retaining rows.
pub(super) fn sk_merge_moments<F: SKFloat>(
    accumulator: &mut SKCovarianceAccumulator<F>,
    batch_count: usize,
    batch_mean: ArrayView1<'_, F>,
    batch_scatter: ArrayView2<'_, F>,
) {
    if batch_count == 0 {
        return;
    }
    if accumulator.samples_seen == 0 {
        accumulator.samples_seen = batch_count;
        accumulator.mean.assign(&batch_mean);
        accumulator.scatter = sk_symmetrized(batch_scatter);
        return;
    }

    let previous_count = accumulator.samples_seen;
    let total_count = previous_count + batch_count;
    let previous_scalar = F::from(previous_count).expect("sample count fits in scalar");
    let batch_scalar = F::from(batch_count).expect("batch count fits in scalar");
    let total_scalar = F::from(total_count).expect("total count fits in scalar");
    let batch_fraction = batch_scalar / total_scalar;
    let cross_weight = (previous_scalar * batch_scalar) / total_scalar;

    let mut delta = batch_mean.to_owned();
    azip!((difference in &mut delta, &running_mean in &accumulator.mean) {
        *difference = *difference - running_mean;
    });
    azip!((running_mean in &mut accumulator.mean, &difference in &delta) {
        *running_mean = *running_mean + difference * batch_fraction;
    });

    accumulator
        .scatter
        .rows_mut()
        .into_iter()
        .zip(batch_scatter.rows())
        .zip(delta.iter())
        .for_each(|((mut scatter_row, batch_row), &row_delta)| {
            azip!((entry in &mut scatter_row, &batch_entry in batch_row, &column_delta in &delta) {
                *entry = *entry + batch_entry + row_delta * column_delta * cross_weight;
            });
        });
    accumulator.scatter = sk_symmetrized(accumulator.scatter.view());
    accumulator.samples_seen = total_count;
}

/// Computes `(n_b, mean_b, scatter_b)` for a dense batch across deterministic
/// 256-row chunks and merges it into `accumulator`.
pub(super) fn sk_update_dense<F: SKFloat>(
    accumulator: &mut SKCovarianceAccumulator<F>,
    backend: &dyn SKMathBackend<F>,
    dense: ArrayView2<'_, F>,
    parallelism: usize,
) {
    let (rows_count, columns_count) = dense.dim();
    if rows_count == 0 || columns_count == 0 {
        return;
    }
    let batch_mean = sk_dense_batch_mean(dense, parallelism);
    let mean_view = batch_mean.view();

    let chunk_scatters: Vec<Array2<F>> = if parallelism > 1 && rows_count > DENSE_MOMENT_ROW_CHUNK {
        dense
            .axis_chunks_iter(Axis(0), DENSE_MOMENT_ROW_CHUNK)
            .into_par_iter()
            .map(|chunk| {
                let centered = sk_center_chunk(chunk, mean_view);
                backend.gemm(centered.t(), centered.view(), F::one(), 1)
            })
            .collect()
    } else {
        dense
            .axis_chunks_iter(Axis(0), DENSE_MOMENT_ROW_CHUNK)
            .map(|chunk| {
                let centered = sk_center_chunk(chunk, mean_view);
                backend.gemm(centered.t(), centered.view(), F::one(), 1)
            })
            .collect()
    };

    let mut batch_scatter = Array2::<F>::zeros((columns_count, columns_count));
    for partial in chunk_scatters {
        azip!((entry in &mut batch_scatter, &value in &partial) {
            *entry = *entry + value;
        });
    }
    sk_merge_moments(
        accumulator,
        rows_count,
        batch_mean.view(),
        batch_scatter.view(),
    );
}

fn sk_dense_batch_mean<F: SKFloat>(dense: ArrayView2<'_, F>, parallelism: usize) -> Array1<F> {
    let (rows_count, columns_count) = dense.dim();
    let rows_scalar = F::from(rows_count).expect("row count fits in scalar");
    let chunk_sums: Vec<Array1<F>> = if parallelism > 1 && rows_count > DENSE_MOMENT_ROW_CHUNK {
        dense
            .axis_chunks_iter(Axis(0), DENSE_MOMENT_ROW_CHUNK)
            .into_par_iter()
            .map(sk_sum_chunk_rows)
            .collect()
    } else {
        dense
            .axis_chunks_iter(Axis(0), DENSE_MOMENT_ROW_CHUNK)
            .map(sk_sum_chunk_rows)
            .collect()
    };
    let mut sums = Array1::<F>::zeros(columns_count);
    for partial in chunk_sums {
        azip!((accumulator in &mut sums, &value in &partial) {
            *accumulator = *accumulator + value;
        });
    }
    sums.mapv(|sum| sum / rows_scalar)
}

fn sk_sum_chunk_rows<F: SKFloat>(chunk: ArrayView2<'_, F>) -> Array1<F> {
    let mut sum = Array1::<F>::zeros(chunk.ncols());
    for row in chunk.rows() {
        azip!((entry in &mut sum, &value in row) {
            *entry = *entry + value;
        });
    }
    sum
}

fn sk_center_chunk<F: SKFloat>(chunk: ArrayView2<'_, F>, mean: ArrayView1<'_, F>) -> Array2<F> {
    let mut centered = chunk.to_owned();
    for mut row in centered.rows_mut() {
        azip!((value in &mut row, &column_mean in mean) {
            *value = *value - column_mean;
        });
    }
    centered
}

pub(super) fn sk_symmetrized<F: SKFloat>(matrix: ArrayView2<'_, F>) -> Array2<F> {
    let half = F::from(0.5).expect("0.5 fits in scalar");
    Array2::from_shape_fn(matrix.dim(), |(row, column)| {
        let average = (matrix[(row, column)] + matrix[(column, row)]) * half;
        if row == column {
            average.max(F::zero())
        } else {
            average
        }
    })
}
