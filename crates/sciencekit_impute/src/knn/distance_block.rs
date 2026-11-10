//! Fixed-size distance blocks for the transform
//! (spec `nearest-neighbor-imputation`).

/// The bytes one distance block may occupy.
///
/// The transform never materializes a full `receivers × training` matrix: it
/// computes the distances of a block of receiver rows at a time, sized from
/// this budget.
pub(super) const SK_DISTANCE_BLOCK_BYTES: usize = 4 * 1024 * 1024;

/// The receiver rows that fit in one distance block for a given training
/// design, always at least one.
pub(super) fn sk_distance_block_rows(
    training_rows: usize,
    scalar_size_bytes: usize,
) -> usize {
    if training_rows == 0 || scalar_size_bytes == 0 {
        return 1;
    }
    (SK_DISTANCE_BLOCK_BYTES / (training_rows * scalar_size_bytes)).max(1)
}
