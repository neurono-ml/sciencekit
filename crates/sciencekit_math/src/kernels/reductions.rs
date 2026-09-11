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
        return Array::zeros(cols);
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
            // Parallel per-row sums: each row reduces independently.
            let sums: Vec<F> = input
                .axis_iter(Axis(1))
                .into_par_iter()
                .map(|row| row.iter().fold(F::zero(), |acc, &value| acc + value))
                .collect();
            Array::from_vec(sums)
        }
        1 => {
            // Sequential per-row sums.
            let mut output = Array::zeros(rows);
            for (index, row) in input.rows().into_iter().enumerate() {
                output[index] = row.iter().fold(F::zero(), |acc, &value| acc + value);
            }
            output
        }
        _ => panic!("sk_axis_sum axis must be 0 or 1, found {axis}"),
    }
}