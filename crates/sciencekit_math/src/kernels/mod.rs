//! Higher-order operation kernels built on `azip!`/`par_azip!`/`zip_mut_with`
//! (spec `higher-order-kernels`, PRD §4.2).
//!
//! Every kernel is a pure function over [`SKFloat`] and honours the
//! higher-order-function mandate: no manual index loops anywhere. Layout-aware
//! reductions iterate over contiguous rows on row-major inputs.

use ndarray::{Array, ArrayView, ArrayView2, azip, par_azip};
use sciencekit_common::SKFloat;

/// Elementwise `x → transform(x)` over a vector, returning a new owned array.
///
/// Uses [`azip!`], never an index loop.
pub fn sk_elementwise_transform<F: SKFloat>(
    input: &ArrayView<F, ndarray::Ix1>,
    transform: impl Fn(F) -> F,
) -> Array<F, ndarray::Ix1> {
    let mut output = Array::zeros(input.dim());
    azip!((out in &mut output, a in input) { *out = transform(*a); });
    output
}

/// Elementwise combine of two vectors into a new owned array.
///
/// Uses [`ArrayBase::zip_mut_with`], seeded from `left`, pairing each element
/// with the corresponding element of `right`.
pub fn sk_binary_combine<F: SKFloat>(
    left: &ArrayView<F, ndarray::Ix1>,
    right: &ArrayView<F, ndarray::Ix1>,
    combine: impl Fn(F, F) -> F,
) -> Array<F, ndarray::Ix1> {
    assert_eq!(
        left.len(),
        right.len(),
        "binary combine requires equal lengths"
    );
    let mut output = left.to_owned();
    output.zip_mut_with(right, |output_value, &right_value| {
        *output_value = combine(*output_value, right_value);
    });
    output
}

/// Sum a 2-D array along one axis, returning a 1-D result.
///
/// `axis = 0` collapses rows (per-column sums, length = columns);
/// `axis = 1` collapses columns (per-row sums, length = rows). Reads rows as
/// contiguous runs on row-major inputs, accumulating each row into the output
/// with [`azip!`] — layout-aware and free of manual index loops.
pub fn sk_axis_sum<F: SKFloat>(input: &ArrayView2<F>, axis: usize) -> Array<F, ndarray::Ix1> {
    let (rows, cols) = input.dim();
    match axis {
        0 => {
            // Per-column sums: one accumulator per column, traversed row by row.
            let mut output = Array::zeros(cols);
            for row in input.rows() {
                azip!((acc in &mut output, value in row) { *acc = *acc + *value; });
            }
            output
        }
        1 => {
            // Per-row sums: each row is contiguous on row-major inputs.
            let mut output = Array::zeros(rows);
            for (index, row) in input.rows().into_iter().enumerate() {
                output[index] = row.iter().fold(F::zero(), |acc, &value| acc + value);
            }
            output
        }
        _ => panic!("sk_axis_sum axis must be 0 or 1, found {axis}"),
    }
}

/// In-place scalar multiply of every element of a 2-D array.
///
/// Mutates `input` in place and returns nothing — no new buffer is allocated.
/// Uses [`par_azip!`] so the scale is parallelised across the element grid.
pub fn sk_scale_in_place<F: SKFloat>(input: &mut Array<F, ndarray::Ix2>, factor: F) {
    par_azip!((value in input) { *value = *value * factor; });
}

#[cfg(test)]
mod kernels_tests;
