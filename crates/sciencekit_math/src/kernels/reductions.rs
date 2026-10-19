//! Reduction kernels: reduce a 2-D array along one axis into a 1-D result.

use ndarray::{Array, ArrayView2, Axis, azip};
use rayon::prelude::*;
use sciencekit_common::SKFloat;
/// Sum a 2-D array along one axis, returning a 1-D result.
///
/// `axis = 0` collapses rows (per-column sums, length = columns);
/// `axis = 1` collapses columns (per-row sums, length = rows).
///
/// Honours the resolved `parallelism`: `1` runs the sequential forms below;
/// `> 1` dispatches across the compute pool. Axis-0 parallelizes rows through
/// per-chunk partial accumulation combined into the output — no concurrent
/// writes to a shared accumulator. Reads rows as contiguous runs on row-major
/// inputs, free of manual index loops.
pub fn sk_axis_sum<F: SKFloat>(
    input: &ArrayView2<F>,
    axis: usize,
    parallelism: usize,
) -> Array<F, ndarray::Ix1> {
    let (rows, cols) = input.dim();
    if rows == 0 {
        // No rows: per-column sums are zero (length columns), per-row sums are
        // empty (length zero).
        return if axis == 1 {
            Array::zeros(0)
        } else {
            Array::zeros(cols)
        };
    }
    match axis {
        0 if parallelism > 1 => {
            // Parallel per-column sums: split rows into chunks, reduce each
            // chunk into its own accumulator, then combine the partials.
            let chunk_rows = rows.div_ceil(parallelism);
            let partials: Vec<Array<F, ndarray::Ix1>> = input
                .axis_chunks_iter(Axis(0), chunk_rows)
                .into_par_iter()
                .map(|chunk| {
                    let mut partial = Array::zeros(cols);
                    for row in chunk.rows() {
                        azip!((acc in &mut partial, value in row) { *acc = *acc + *value; });
                    }
                    partial
                })
                .collect();
            let mut output = Array::zeros(cols);
            for partial in partials {
                azip!((out in &mut output, value in &partial) { *out = *out + *value; });
            }
            output
        }
        0 => {
            // Sequential per-column sums: one accumulator per column.
            let mut output = Array::zeros(cols);
            for row in input.rows() {
                azip!((acc in &mut output, value in row) { *acc = *acc + *value; });
            }
            output
        }
        1 if parallelism > 1 => {
            // Parallel per-row sums: each row reduces independently. Iterating
            // `Axis(0)` yields the rows themselves (length = rows).
            let sums: Vec<F> = input
                .axis_iter(Axis(0))
                .into_par_iter()
                .map(|row| row.iter().fold(F::zero(), |acc, &value| acc + value))
                .collect();
            Array::from_vec(sums)
        }
        1 => {
            // Sequential per-row sums, one reduction per row.
            input
                .rows()
                .into_iter()
                .map(|row| row.iter().fold(F::zero(), |acc, &value| acc + value))
                .collect()
        }
        _ => panic!("sk_axis_sum axis must be 0 or 1, found {axis}"),
    }
}

/// Sum of squares of a 2-D array along one axis, returning a 1-D result.
///
/// `axis = 0` collapses rows (per-column Σx², length = columns);
/// `axis = 1` collapses columns (per-row Σx², length = rows).
///
/// The companion to [`sk_axis_sum`] for variance-style statistics
/// (mean from Σx, variance from Σx²): same parallelism contract — `1` runs
/// the sequential forms, `> 1` dispatches across the compute pool through
/// per-chunk partial accumulators combined once at the end, free of manual
/// index loops. Shares the [`SK_AXIS_SUM_GRAIN`](super::SK_AXIS_SUM_GRAIN)
/// reduction grain.
pub fn sk_axis_sum_squares<F: SKFloat>(
    input: &ArrayView2<F>,
    axis: usize,
    parallelism: usize,
) -> Array<F, ndarray::Ix1> {
    let (rows, cols) = input.dim();
    match axis {
        0 if parallelism > 1 => {
            let chunk_rows = rows.div_ceil(parallelism);
            let partials: Vec<Array<F, ndarray::Ix1>> = input
                .axis_chunks_iter(Axis(0), chunk_rows.max(1))
                .into_par_iter()
                .map(|chunk| sum_squares_sequential(&chunk.view(), 0))
                .collect();
            combine_partials(partials, cols)
        }
        0 => sum_squares_sequential(input, 0),
        1 if parallelism > 1 => {
            // Parallel per-row Σx²: iterating `Axis(0)` yields the rows
            // themselves, so the result has one element per row.
            let squares: Vec<F> = input
                .axis_iter(Axis(0))
                .into_par_iter()
                .map(|row| {
                    row.iter()
                        .fold(F::zero(), |acc, &value| acc + value * value)
                })
                .collect();
            Array::from_vec(squares)
        }
        1 => sum_squares_sequential(input, 1),
        _ => panic!("sk_axis_sum_squares axis must be 0 or 1, found {axis}"),
    }
}

/// Sequential Σx² along one axis over contiguous row runs.
fn sum_squares_sequential<F: SKFloat>(
    input: &ArrayView2<F>,
    axis: usize,
) -> Array<F, ndarray::Ix1> {
    let (_, cols) = input.dim();
    if axis == 0 {
        let mut output = Array::zeros(cols);
        for row in input.rows() {
            azip!((acc in &mut output, value in row) { *acc = *acc + *value * *value; });
        }
        output
    } else {
        input
            .rows()
            .into_iter()
            .map(|row| {
                row.iter()
                    .fold(F::zero(), |acc, &value| acc + value * value)
            })
            .collect()
    }
}

/// Combine per-worker partial accumulators into a single result.
fn combine_partials<F: SKFloat>(
    partials: Vec<Array<F, ndarray::Ix1>>,
    len: usize,
) -> Array<F, ndarray::Ix1> {
    let mut output = Array::zeros(len);
    for partial in partials {
        azip!((out in &mut output, value in &partial) { *out = *out + *value; });
    }
    output
}
