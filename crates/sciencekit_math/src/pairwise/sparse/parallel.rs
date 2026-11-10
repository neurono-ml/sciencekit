//! Row-parallel fill helper shared by the sparse distance kernels.

use ndarray::{Array, Array2, ArrayViewMut1, Axis};
use rayon::prelude::*;
use sciencekit_common::SKFloat;

/// Pair count below which a sparse distance matrix is filled sequentially.
///
/// This is the same grain the dense kernels use for their row-parallel paths.
pub(crate) const SK_SPARSE_PAIRWISE_PARALLEL_GRAIN: usize = 1024;

/// Fill a distance matrix row by row, sharing the work across rayon threads.
///
/// Each row writes into its own output row, so the parallel path needs no locks
/// and is deterministic: every row is computed independently and identically to
/// the sequential path.
pub(crate) fn sk_fill_distance_matrix<F, C>(
    left_rows: usize,
    right_rows: usize,
    compute_row: C,
) -> Array2<F>
where
    F: SKFloat,
    C: Fn(usize, ArrayViewMut1<'_, F>) + Sync + Send,
{
    let mut output = Array::zeros((left_rows, right_rows));
    if left_rows * right_rows < SK_SPARSE_PAIRWISE_PARALLEL_GRAIN {
        output
            .axis_iter_mut(Axis(0))
            .enumerate()
            .for_each(|(i, row)| compute_row(i, row));
    } else {
        output
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, row)| compute_row(i, row));
    }
    output
}
