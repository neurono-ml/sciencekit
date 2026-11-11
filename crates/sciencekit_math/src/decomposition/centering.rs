//! Implicit centering of the randomized range finder.
//!
//! A centering vector `μ` (one mean per feature) defines the centered matrix
//! `Xc = X − 1 μᵀ`. The kernel never materializes `Xc`: the rank-one
//! correction is folded into the projected products instead.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sciencekit_common::SKFloat;

/// Subtracts the sample-side centering contribution `1 (μᵀ D)` from `target`.
///
/// `target` holds `m × p` values (one row per observation) and `D` the dense
/// `n × p` product the contribution is derived from.
pub(crate) fn sk_subtract_sample_mean_contribution<F: SKFloat>(
    target: &mut Array2<F>,
    mean: &ArrayView1<'_, F>,
    product: &ArrayView2<'_, F>,
) {
    let mut contribution = Array1::<F>::zeros(product.ncols());
    for (column, value) in contribution.iter_mut().enumerate() {
        for (feature, &mean_value) in mean.iter().enumerate() {
            *value = *value + mean_value * product[(feature, column)];
        }
    }
    for mut row in target.rows_mut() {
        for (index, &amount) in contribution.iter().enumerate() {
            row[index] = row[index] - amount;
        }
    }
}

/// Subtracts the feature-side centering contribution `μ (1ᵀ D)` from `target`.
///
/// `target` holds `n × p` values (one row per feature) and `D` the dense
/// `m × p` product the contribution is derived from.
pub(crate) fn sk_subtract_feature_mean_contribution<F: SKFloat>(
    target: &mut Array2<F>,
    mean: &ArrayView1<'_, F>,
    product: &ArrayView2<'_, F>,
) {
    let mut contribution = Array1::<F>::zeros(product.ncols());
    for (column, value) in contribution.iter_mut().enumerate() {
        for row in 0..product.nrows() {
            *value = *value + product[(row, column)];
        }
    }
    for (feature, mut row) in target.rows_mut().into_iter().enumerate() {
        let mean_value = mean[feature];
        for (index, &amount) in contribution.iter().enumerate() {
            row[index] = row[index] - mean_value * amount;
        }
    }
}
