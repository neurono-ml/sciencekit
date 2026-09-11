//! Scaling kernels: in-place scalar transforms over whole arrays.

use ndarray::{Array, azip, par_azip};
use sciencekit_common::SKFloat;

/// In-place scalar multiply of every element of a 2-D array.
///
/// Mutates `input` in place and returns nothing — no new buffer is allocated.
/// Honours the resolved `parallelism`: `1` runs the sequential [`azip!`] form,
/// `> 1` dispatches [`par_azip!`] across the element grid.
pub fn sk_scale_in_place<F: SKFloat>(input: &mut Array<F, ndarray::Ix2>, factor: F, parallelism: usize) {
    if parallelism > 1 {
        par_azip!((value in input) { *value = *value * factor; });
    } else {
        azip!((value in input) { *value = *value * factor; });
    }
}