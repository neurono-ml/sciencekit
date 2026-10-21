//! Conjugate-gradient solve of a symmetric positive-definite system (spec
//! `iterative-linear-solvers`).

use ndarray::{Array1, ArrayView1, azip};
use sciencekit_common::{SKError, SKFloat};

use super::SKLinearOperator;

/// Solve `operator · x = right_hand_side` for a symmetric positive-definite
/// `operator` with the conjugate-gradient method.
///
/// The iteration starts from a zero guess and stops as soon as the residual
/// 2-norm falls to or below `tolerance`, or the residual vanishes exactly. When
/// the iteration budget is exhausted before either happens it returns
/// [`SKError::NotConverged`] carrying the number of iterations executed — never
/// a stale vector passed off as a solution. A zero budget therefore fails
/// immediately with `NotConverged { iterations: 0 }`.
///
/// The iteration is a sequential recurrence; the operator's own application is
/// where any parallelism lives. Both `right_hand_side` and the result are
/// `operator.dimension()` long.
pub fn sk_conjugate_gradient<F: SKFloat, O: SKLinearOperator<F>>(
    operator: &O,
    right_hand_side: ArrayView1<F>,
    tolerance: F,
    maximum_number_of_iterations: usize,
) -> Result<Array1<F>, SKError> {
    let mut solution = Array1::zeros(operator.dimension());
    let mut residual = right_hand_side.to_owned();
    let mut direction = residual.clone();
    let mut residual_squared = residual.dot(&residual);

    if maximum_number_of_iterations == 0 {
        return Err(SKError::not_converged(0));
    }
    if converged(residual_squared, tolerance) {
        return Ok(solution);
    }

    for _ in 0..maximum_number_of_iterations {
        let applied = operator.apply(direction.view());
        let curvature = direction.dot(&applied);
        let step = residual_squared / curvature;
        azip!((solution in &mut solution, direction in &direction) {
            *solution = *solution + step * *direction;
        });
        azip!((residual in &mut residual, applied in &applied) {
            *residual = *residual - step * *applied;
        });
        let residual_squared_new = residual.dot(&residual);
        if converged(residual_squared_new, tolerance) {
            return Ok(solution);
        }
        let momentum = residual_squared_new / residual_squared;
        azip!((direction in &mut direction, residual in &residual) {
            *direction = *residual + momentum * *direction;
        });
        residual_squared = residual_squared_new;
    }

    Err(SKError::not_converged(maximum_number_of_iterations))
}

/// Whether the residual 2-norm has fallen to or below the tolerance, or the
/// residual has vanished exactly.
fn converged<F: SKFloat>(residual_squared: F, tolerance: F) -> bool {
    residual_squared == F::zero() || residual_squared.sqrt() <= tolerance
}
