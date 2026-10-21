//! Centered-design helpers shared by the dense ridge solve paths
//! (spec `ridge-regression`).

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, azip};
use sciencekit_common::SKFloat;

/// Subtract the feature means from every row of the design.
pub(super) fn center_design<F: SKFloat>(dense: ArrayView2<F>, means: ArrayView1<F>) -> Array2<F> {
    let mut centered = dense.to_owned();
    centered.axis_iter_mut(Axis(0)).for_each(|mut row| {
        azip!((value in &mut row, mean in &means) { *value = *value - *mean; });
    });
    centered
}

/// Subtract the target mean from every target entry.
pub(super) fn center_target<F: SKFloat>(target: ArrayView1<F>, mean: F) -> Array1<F> {
    target.mapv(|value| value - mean)
}
