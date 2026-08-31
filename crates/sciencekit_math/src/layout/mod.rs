//! Memory-layout helpers (spec `layout`, PRD §4.3).
//!
//! Hot kernels want contiguous inputs. These helpers detect the layout of an
//! [`ArrayView`] and produce a contiguous owner only when needed, so calling
//! code can feed zero-copy views into SIMD loops without surprises.

use ndarray::{ArrayView, CowArray};
use sciencekit_common::SKFloat;

/// The memory layout of a 2-D array, as seen by an algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKMemoryLayout {
    /// Row-major (C) contiguous: rows are stored back to back.
    /// Column-major (F) contiguous: columns are stored back to back.
    /// Contiguous and consecutive, neither pure C nor pure F.
    CContiguous,
    /// Column-major (F) contiguous.
    FContiguous,
    /// Neither C- nor F-contiguous: rows and columns are strided.
    Strided,
}

/// Detect the memory layout of a 2-D array.
///
/// Prefers contiguous over strided and is exact for the standard layouts:
/// C-contiguous first (row-major), then F-contiguous, then `Strided`.
pub fn sk_memory_layout<F>(array: &ArrayView<F, ndarray::Ix2>) -> SKMemoryLayout {
    if array.is_standard_layout() {
        SKMemoryLayout::CContiguous
    } else if array.t().is_standard_layout() {
        SKMemoryLayout::FContiguous
    } else {
        SKMemoryLayout::Strided
    }
}

/// Report whether a 2-D array is C-contiguous (row-major).
pub fn sk_is_c_contiguous<F>(array: &ArrayView<F, ndarray::Ix2>) -> bool {
    array.is_standard_layout()
}

/// Report whether a 2-D array is F-contiguous (column-major).
pub fn sk_is_f_contiguous<F>(array: &ArrayView<F, ndarray::Ix2>) -> bool {
    array.t().is_standard_layout()
}

/// Force contiguity, borrowing when already contiguous and owning otherwise.
///
/// Returns a [`CowArray`]: when `array` is already contiguous this is a
/// zero-copy borrow of the input (a no-op); when it is strided it owns a fresh
/// contiguous copy via `to_owned`. Callers can treat the result as an
/// [`ArrayView`] and feed it into layout-sensitive kernels.
pub fn sk_force_contiguous<'a, F: SKFloat>(
    array: &'a ArrayView<F, ndarray::Ix2>,
) -> CowArray<'a, F, ndarray::Ix2> {
    if sk_is_c_contiguous(array) || sk_is_f_contiguous(array) {
        CowArray::from(array.view())
    } else {
        CowArray::from(array.to_owned())
    }
}

#[cfg(test)]
mod layout_tests;
