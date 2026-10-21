//! The Cholesky path: solve the centered normal equations through the backend's
//! Cholesky factor (spec `ridge-regression`).

use ndarray::{Array1, ArrayView1, ArrayView2, Axis};
use sciencekit_common::{SKBackendKind, SKError};
use sciencekit_math::SKBackendSelection;

/// Solve `gram_plus_alpha · β = cross` through the backend's Cholesky factor.
///
/// The factor is `A = L Lᵀ`; the system is solved by the two triangular passes
/// `L y = cross` then `Lᵀ β = y`.
pub(crate) fn solve_with_cholesky<F: SKBackendSelection>(
    backend: SKBackendKind,
    gram_plus_alpha: ArrayView2<F>,
    cross: ArrayView1<F>,
) -> Result<Array1<F>, SKError> {
    let machine = F::select_math_backend(backend);
    let factor = machine.cholesky(gram_plus_alpha)?;
    let right_hand_side = cross.insert_axis(Axis(1));
    let intermediate = machine.solve_triangular(factor.view(), right_hand_side, true, false)?;
    let solution = machine.solve_triangular(factor.t(), intermediate.view(), false, false)?;
    Ok(solution.column(0).to_owned())
}
