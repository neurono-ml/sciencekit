//! Fixed-size distance blocks for the transform
//! (spec `nearest-neighbor-imputation`).

use std::ops::Range;

use ndarray::{Array2, ArrayView2};
use sciencekit_common::SKFloat;
use sciencekit_math::sk_nan_euclidean_distance_matrix;

/// The bytes one distance block may occupy.
///
/// The transform never materializes a full `receivers × training` matrix: it
/// computes the distances of a block of receiver rows at a time, sized from
/// this budget.
pub(super) const SK_DISTANCE_BLOCK_BYTES: usize = 4 * 1024 * 1024;

/// The receiver rows that fit in one distance block for a given training
/// design, always at least one.
pub(super) fn sk_distance_block_rows(training_rows: usize, scalar_size_bytes: usize) -> usize {
    if training_rows == 0 || scalar_size_bytes == 0 {
        return 1;
    }
    (SK_DISTANCE_BLOCK_BYTES / (training_rows * scalar_size_bytes)).max(1)
}

/// The contiguous receiver-row ranges of the distance plan.
///
/// The ranges partition `0..rows_count` without gaps or overlap, so every
/// receiver row is processed exactly once regardless of the block width.
pub(super) fn sk_distance_block_ranges(rows_count: usize, block_rows: usize) -> Vec<Range<usize>> {
    let block_rows = block_rows.max(1);
    (0..rows_count)
        .step_by(block_rows)
        .map(|start| start..(start + block_rows).min(rows_count))
        .collect()
}

/// The NaN-aware distances from a block of receiver rows to every training
/// row.
pub(super) fn sk_distance_block<F: SKFloat>(
    receivers: &ArrayView2<'_, F>,
    training: &ArrayView2<'_, F>,
) -> Array2<F> {
    sk_nan_euclidean_distance_matrix(receivers, training)
}
