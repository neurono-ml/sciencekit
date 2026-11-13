//! Test-only instrumentation of the dense intermediates of the kernel.
//!
//! The randomized range finder keeps every dense intermediate small: the
//! products of the range search are `m × (k + oversampling)` and
//! `n × (k + oversampling)`. The recorder proves that a sparse input never
//! materializes a dense copy of the matrix.

use std::cell::Cell;

thread_local! {
    /// The largest dense intermediate (in cells) produced on the current
    /// thread.
    static DENSE_INTERMEDIATE_CELLS: Cell<usize> = const { Cell::new(0) };
}

/// Records the size of a dense intermediate.
pub(crate) fn sk_record_dense_intermediate(cells: usize) {
    DENSE_INTERMEDIATE_CELLS.with(|cell| cell.set(cell.get().max(cells)));
}

/// The largest dense intermediate (in cells) recorded on the current thread.
pub(crate) fn sk_largest_dense_intermediate() -> usize {
    DENSE_INTERMEDIATE_CELLS.with(|cell| cell.get())
}

/// Resets the recorded maximum on the current thread.
pub(crate) fn sk_reset_dense_intermediate() {
    DENSE_INTERMEDIATE_CELLS.with(|cell| cell.set(0));
}
