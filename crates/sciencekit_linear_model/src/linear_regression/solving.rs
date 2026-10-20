//! Backend dispatch for the least-squares solve (spec `backend-selection`).

use ndarray::{ArrayView1, ArrayView2, Axis};
use sciencekit_common::{SKBackendKind, SKError, SKFloat};
use sciencekit_math::{
    SKFaerBackend, SKLeastSquaresSolution, SKMathBackend, SKMatrixMultiplyBackend,
    sk_default_math_backend,
};

/// Solve `design · x ≈ target` in the least-squares sense through the backend
/// matching the configured kind, at operation time (never in `build()`).
///
/// The solve is the backend's least-squares kernel — the SVD/pseudoinverse path
/// — so the solution is the minimum-norm one under rank deficiency and the rank
/// and singular values come back for the model to surface.
pub(super) fn solve_least_squares<F: SKFloat>(
    kind: SKBackendKind,
    design: ArrayView2<F>,
    target: ArrayView1<F>,
) -> Result<SKLeastSquaresSolution<F>, SKError>
where
    SKFaerBackend: SKMathBackend<F>,
    SKMatrixMultiplyBackend: SKMathBackend<F>,
{
    let right_hand_side = target.insert_axis(Axis(1));
    match kind {
        SKBackendKind::Faer => SKFaerBackend::new().lstsq(design, right_hand_side),
        SKBackendKind::MatrixMultiply => {
            SKMatrixMultiplyBackend::new().lstsq(design, right_hand_side)
        }
        SKBackendKind::NdArrayLinalg => {
            sk_default_math_backend::<F>().lstsq(design, right_hand_side)
        }
    }
}
