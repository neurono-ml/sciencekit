//! Per-axis min/max extremum reduction: the order-independent companion of the
//! sum reductions.

use ndarray::{Array, ArrayView1, ArrayView2, Axis, Ix1, azip};
use rayon::prelude::*;
use sciencekit_common::SKFloat;

/// Compute the per-axis minimum and the per-axis maximum of a 2-D array in one
/// pass, returning the two 1-D results.
///
/// `axis = 0` collapses rows (per-column extrema, length = columns);
/// `axis = 1` collapses columns (per-row extrema, length = rows).
///
/// Honours the resolved `parallelism`: `1` runs the sequential forms below;
/// `> 1` dispatches across the compute pool through per-chunk partial extrema
/// combined once at the end — no concurrent writes to a shared accumulator.
/// Min/max are order-independent, so the parallel result equals the sequential
/// one exactly. Shares the [`SK_AXIS_SUM_GRAIN`](super::SK_AXIS_SUM_GRAIN)
/// reduction grain.
pub fn sk_axis_min_max<F: SKFloat>(
    input: &ArrayView2<F>,
    axis: usize,
    parallelism: usize,
) -> (Array<F, Ix1>, Array<F, Ix1>) {
    let (rows, cols) = input.dim();
    match axis {
        0 if parallelism > 1 => {
            // Parallel per-column extrema: split rows into chunks, reduce each
            // chunk into its own partial, then combine the partials.
            let chunk_rows = rows.div_ceil(parallelism).max(1);
            let partials: Vec<(Array<F, Ix1>, Array<F, Ix1>)> = input
                .axis_chunks_iter(Axis(0), chunk_rows)
                .into_par_iter()
                .map(|chunk| min_max_sequential(&chunk.view(), 0))
                .collect();
            combine_partials(partials, cols)
        }
        0 => min_max_sequential(input, 0),
        1 if parallelism > 1 => {
            // Parallel per-row extrema: each row reduces independently.
            let pairs: Vec<(F, F)> = input
                .axis_iter(Axis(0))
                .into_par_iter()
                .map(|row| row_extremes(&row))
                .collect();
            split_pairs(pairs)
        }
        1 => min_max_sequential(input, 1),
        _ => panic!("sk_axis_min_max axis must be 0 or 1, found {axis}"),
    }
}

/// Sequential extrema along one axis over contiguous row runs.
fn min_max_sequential<F: SKFloat>(
    input: &ArrayView2<F>,
    axis: usize,
) -> (Array<F, Ix1>, Array<F, Ix1>) {
    if axis == 0 {
        let (_, cols) = input.dim();
        let mut minima = Array::from_elem(cols, F::infinity());
        let mut maxima = Array::from_elem(cols, F::neg_infinity());
        for row in input.rows() {
            azip!((minimum in &mut minima, maximum in &mut maxima, value in row) {
                *minimum = minimum.min(*value);
                *maximum = maximum.max(*value);
            });
        }
        (minima, maxima)
    } else {
        let pairs: Vec<(F, F)> = input
            .rows()
            .into_iter()
            .map(|row| row_extremes(&row))
            .collect();
        split_pairs(pairs)
    }
}

/// The minimum and maximum of one row, folded in a single pass.
fn row_extremes<F: SKFloat>(row: &ArrayView1<F>) -> (F, F) {
    row.iter().fold(
        (F::infinity(), F::neg_infinity()),
        |(minimum, maximum), &value| (minimum.min(value), maximum.max(value)),
    )
}

/// Split per-axis `(minimum, maximum)` pairs into the two output vectors.
fn split_pairs<F: SKFloat>(pairs: Vec<(F, F)>) -> (Array<F, Ix1>, Array<F, Ix1>) {
    let minima = pairs.iter().map(|&(minimum, _)| minimum).collect();
    let maxima = pairs.iter().map(|&(_, maximum)| maximum).collect();
    (minima, maxima)
}

/// Combine per-worker partial extrema into a single result (element-wise min of
/// the minima and max of the maxima).
fn combine_partials<F: SKFloat>(
    partials: Vec<(Array<F, Ix1>, Array<F, Ix1>)>,
    len: usize,
) -> (Array<F, Ix1>, Array<F, Ix1>) {
    let mut minima = Array::from_elem(len, F::infinity());
    let mut maxima = Array::from_elem(len, F::neg_infinity());
    for (partial_minima, partial_maxima) in partials {
        azip!((minimum in &mut minima, value in &partial_minima) {
            *minimum = minimum.min(*value);
        });
        azip!((maximum in &mut maxima, value in &partial_maxima) {
            *maximum = maximum.max(*value);
        });
    }
    (minima, maxima)
}
