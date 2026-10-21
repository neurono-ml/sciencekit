//! Backend dispatch for the least-squares solve (spec `backend-selection`).

use ndarray::{ArrayView1, ArrayView2, Axis};
use sciencekit_common::{SKBackendKind, SKError};
use sciencekit_math::{SKBackendSelection, SKLeastSquaresSolution};

/// Solve `design · x ≈ target` in the least-squares sense through the backend
/// matching the configured kind, at operation time (never in `build()`).
///
/// The solve is the backend's least-squares kernel — the SVD/pseudoinverse path
/// — so the solution is the minimum-norm one under rank deficiency and the rank
/// and singular values come back for the model to surface.
///
/// The kind→backend mapping lives in `sciencekit_math` (via
/// [`SKBackendSelection`]), so the feature-gated `blas-backend` configuration is
/// resolved inside the crate that owns the feature and never leaks a bound here.
pub(super) fn solve_least_squares<F: SKBackendSelection>(
    kind: SKBackendKind,
    design: ArrayView2<F>,
    target: ArrayView1<F>,
) -> Result<SKLeastSquaresSolution<F>, SKError> {
    let right_hand_side = target.insert_axis(Axis(1));
    F::select_math_backend(kind).lstsq(design, right_hand_side)
}
