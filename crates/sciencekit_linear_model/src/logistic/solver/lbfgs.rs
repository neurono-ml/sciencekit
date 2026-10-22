//! Limited-memory BFGS with an Armijo backtracking line search (spec
//! `logistic-solver`, Decision 3).
//!
//! The generic minimizer is solver-agnostic: callers pass an `evaluate`
//! closure returning the objective value and writing its gradient. The two-loop
//! recursion keeps a bounded curvature history, so memory stays `O(history ·
//! dimension)` instead of `O(dimension²)`.

use ndarray::{Array1, ArrayView1, ArrayViewMut1};
use sciencekit_common::{SKError, SKFloat};

/// The L-BFGS convergence controls.
pub(crate) struct SKLbfgsConfig<F> {
    /// The convergence tolerance on the gradient infinity norm.
    pub(crate) tolerance: F,
    /// The iteration budget.
    pub(crate) maximum_number_of_iterations: usize,
    /// The curvature-history length.
    pub(crate) history_size: usize,
}

/// Minimize a differentiable objective from `point`, returning the minimizer
/// and the iterations spent.
///
/// Converges when the gradient infinity norm falls to `tolerance`; exhausting
/// the budget raises [`SKError::NotConverged`].
pub(crate) fn sk_lbfgs_minimize<F, O>(
    mut evaluate: O,
    mut point: Array1<F>,
    config: &SKLbfgsConfig<F>,
) -> Result<(Array1<F>, usize), SKError>
where
    F: SKFloat,
    O: FnMut(ArrayView1<F>, ArrayViewMut1<F>) -> F,
{
    let dimension = point.len();
    let mut gradient = Array1::<F>::zeros(dimension);
    let mut value = evaluate(point.view(), gradient.view_mut());
    let mut s_history: Vec<Array1<F>> = Vec::new();
    let mut y_history: Vec<Array1<F>> = Vec::new();
    let mut rho_history: Vec<F> = Vec::new();
    let half = F::from(0.5).expect("0.5 fits");
    let armijo = F::from(1e-4).expect("1e-4 fits");
    let minimum_step = F::from(1e-18).expect("fits");

    for iteration in 0..config.maximum_number_of_iterations {
        if sk_max_abs(gradient.view()) <= config.tolerance {
            return Ok((point, iteration));
        }
        let direction = sk_search_direction(gradient.view(), &s_history, &y_history, &rho_history);
        let slope = sk_dot(gradient.view(), direction.view());
        let mut step = F::one();
        let (new_point, new_value, new_gradient) = loop {
            let candidate = &point + &direction.mapv(|value| value * step);
            let mut candidate_gradient = Array1::<F>::zeros(dimension);
            let candidate_value = evaluate(candidate.view(), candidate_gradient.view_mut());
            if candidate_value <= value + armijo * step * slope || step <= minimum_step {
                break (candidate, candidate_value, candidate_gradient);
            }
            step = step * half;
        };
        let s = &new_point - &point;
        let y = &new_gradient - &gradient;
        let curvature = sk_dot(s.view(), y.view());
        if curvature > F::zero() {
            if s_history.len() == config.history_size {
                s_history.remove(0);
                y_history.remove(0);
                rho_history.remove(0);
            }
            s_history.push(s);
            y_history.push(y);
            rho_history.push(F::one() / curvature);
        }
        point = new_point;
        value = new_value;
        gradient = new_gradient;
    }
    Err(SKError::not_converged(config.maximum_number_of_iterations))
}

/// The L-BFGS two-loop recursion producing a descent direction.
fn sk_search_direction<F: SKFloat>(
    gradient: ArrayView1<F>,
    s_history: &[Array1<F>],
    y_history: &[Array1<F>],
    rho_history: &[F],
) -> Array1<F> {
    let length = s_history.len();
    let mut q = gradient.to_owned();
    let mut alpha = vec![F::zero(); length];
    for index in (0..length).rev() {
        let coefficient = rho_history[index] * sk_dot(s_history[index].view(), q.view());
        alpha[index] = coefficient;
        sk_axpy(-coefficient, y_history[index].view(), q.view_mut());
    }
    if length > 0 {
        let last = length - 1;
        let yy = sk_dot(y_history[last].view(), y_history[last].view());
        let sy = sk_dot(s_history[last].view(), y_history[last].view());
        let scale = if yy > F::zero() { sy / yy } else { F::one() };
        q.mapv_inplace(|value| value * scale);
    }
    for index in 0..length {
        let coefficient = rho_history[index] * sk_dot(y_history[index].view(), q.view());
        sk_axpy(
            alpha[index] - coefficient,
            s_history[index].view(),
            q.view_mut(),
        );
    }
    q.mapv(|value| -value)
}

/// `y = y + scale · x`.
fn sk_axpy<F: SKFloat>(scale: F, x: ArrayView1<F>, mut y: ArrayViewMut1<F>) {
    y.iter_mut()
        .zip(x)
        .for_each(|(entry, &value)| *entry = *entry + scale * value);
}

/// The inner product of two vectors.
pub(crate) fn sk_dot<F: SKFloat>(left: ArrayView1<F>, right: ArrayView1<F>) -> F {
    left.iter()
        .zip(right)
        .fold(F::zero(), |accumulator, (&a, &b)| accumulator + a * b)
}

/// The infinity norm of a vector.
pub(crate) fn sk_max_abs<F: SKFloat>(vector: ArrayView1<F>) -> F {
    vector
        .iter()
        .fold(F::zero(), |accumulator, value| accumulator.max(value.abs()))
}
