//! The dense missingness mask of the nearest-neighbor imputer
//! (spec `nearest-neighbor-imputation`).
//!
//! The marker predicate is the one shared with `SKSimpleImputer` and
//! `SKMissingIndicator`: `NaN` matches through `is_nan`, any other scalar
//! matches through equality. This module only turns that predicate into the
//! dense `rows × columns` mask the fit retains.

use ndarray::{Array2, ArrayView2};
use sciencekit_common::SKFloat;

use crate::missingness::sk_is_missing;

/// The training design with every missing cell replaced by `NaN`.
///
/// The NaN-aware Euclidean kernel ignores a coordinate exactly when either
/// side is `NaN`. A scalar marker such as `-999` has to be normalized before
/// the kernel runs, otherwise the sentinel would count as an observed value.
/// Observed cells are copied unchanged, so donor values stay exact.
pub(super) fn sk_canonicalize_missing<F: SKFloat>(
    features: &ArrayView2<'_, F>,
    marker: F,
) -> Array2<F> {
    Array2::from_shape_fn(features.dim(), |(row, column)| {
        let value = features[[row, column]];
        if sk_is_missing(value, marker) {
            F::nan()
        } else {
            value
        }
    })
}

/// A `rows × columns` mask with `true` exactly at the cells the marker
/// identifies as missing.
pub(super) fn sk_dense_missing_mask<F: SKFloat>(
    features: &ArrayView2<'_, F>,
    marker: F,
) -> Array2<bool> {
    Array2::from_shape_fn(features.dim(), |(row, column)| {
        sk_is_missing(features[[row, column]], marker)
    })
}
