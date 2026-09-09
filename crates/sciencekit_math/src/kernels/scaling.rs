//! Scaling kernels: in-place scalar transforms over whole arrays.

use ndarray::{Array, par_azip};
use sciencekit_common::SKFloat;

/// In-place scalar multiply of every element of a 2-D array.
///
/// Mutates `input` in place and returns nothing — no new buffer is allocated.
/// Uses [`par_azip!`] so the scale is parallelised across the element grid.
pub fn sk_scale_in_place<F: SKFloat>(input: &mut Array<F, ndarray::Ix2>, factor: F) {
    par_azip!((value in input) { *value = *value * factor; });
}
