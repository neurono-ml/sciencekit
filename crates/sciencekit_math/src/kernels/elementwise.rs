//! Elementwise higher-order kernels: `x → transform(x)` and pairwise combine.

use ndarray::{Array, ArrayView, azip, par_azip};
use sciencekit_common::SKFloat;

/// Elementwise `x → transform(x)` over a vector, returning a new owned array.
///
/// Honours the resolved `parallelism`: `1` runs the sequential [`azip!`] form
/// with zero dispatch overhead; `> 1` dispatches [`par_azip!`] across the
/// compute pool. Uses an iterator form, never a manual index loop.
pub fn sk_elementwise_transform<F: SKFloat>(
    input: &ArrayView<F, ndarray::Ix1>,
    transform: impl Fn(F) -> F + Sync + Send,
    parallelism: usize,
) -> Array<F, ndarray::Ix1> {
    let mut output = Array::zeros(input.dim());
    if parallelism > 1 {
        par_azip!((out in &mut output, a in input) { *out = transform(*a); });
    } else {
        azip!((out in &mut output, a in input) { *out = transform(*a); });
    }
    output
}

/// Elementwise combine of two vectors into a new owned array.
///
/// Honours the resolved `parallelism`: `1` runs the sequential
/// [`ArrayBase::zip_mut_with`] form; `> 1` dispatches [`par_azip!`] across the
/// compute pool. Seeded from `left`, pairing each element with the
/// corresponding element of `right`.
pub fn sk_binary_combine<F: SKFloat>(
    left: &ArrayView<F, ndarray::Ix1>,
    right: &ArrayView<F, ndarray::Ix1>,
    combine: impl Fn(F, F) -> F + Sync + Send,
    parallelism: usize,
) -> Array<F, ndarray::Ix1> {
    assert_eq!(
        left.len(),
        right.len(),
        "binary combine requires equal lengths"
    );
    let mut output = left.to_owned();
    if parallelism > 1 {
        par_azip!((out in &mut output, l in left, r in right) { *out = combine(*l, *r); });
    } else {
        output.zip_mut_with(right, |output_value, &right_value| {
            *output_value = combine(*output_value, right_value);
        });
    }
    output
}