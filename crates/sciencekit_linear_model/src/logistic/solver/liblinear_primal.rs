//! The LIBLINEAR L1-regularized primal coordinate descent for a binary head
//! (spec `logistic-solver`, Decision 3).
//!
//! Each coordinate minimizes the logistic loss plus the L1 penalty under a
//! diagonal quadratic upper bound (a soft-threshold step), keeping the design
//! sparse; as in scikit-learn's LIBLINEAR path the intercept is regularised.

use ndarray::Array1;
use sciencekit_common::{SKError, SKFloat};

use crate::logistic::config::SKLogisticConfig;
use crate::logistic::design::SKLogisticDesign;
use crate::logistic::objective::sk_sigmoid;

/// Solve a binary head with the LIBLINEAR L1 primal coordinate descent.
pub(crate) fn sk_solve_binary_head_liblinear_l1<F, D>(
    design: &D,
    labels: &[F],
    weights: &[F],
    config: &SKLogisticConfig<F>,
) -> Result<(Array1<F>, F, usize), SKError>
where
    F: SKFloat,
    D: SKLogisticDesign<F> + Sync,
{
    let columns = design.columns_count();
    let rows = design.rows_count();
    let inverse_rows = F::one() / F::from(rows).expect("row counts fit");
    let quarter = F::from(0.25).expect("fits");
    let mut w = Array1::<F>::zeros(columns);
    let mut intercept = F::zero();
    let mut linear = Array1::<F>::zeros(rows);

    let mut iterations = 0usize;
    for sweep in 0..config.maximum_number_of_iterations {
        let mut maximum_change = F::zero();
        for column in 0..columns {
            let residual = sk_residual(&linear, labels, weights, inverse_rows);
            let gradient = design.column_dot(column, residual.view());
            let curvature = design
                .weighted_column_squared_norm(column, ndarray::ArrayView1::from(weights))
                * inverse_rows
                * quarter;
            if curvature <= F::zero() {
                continue;
            }
            let new_value =
                sk_soft_threshold(w[column] - gradient / curvature, config.alpha / curvature);
            let change = new_value - w[column];
            if change != F::zero() {
                w[column] = new_value;
                design.add_column_scaled(column, change, linear.view_mut());
                maximum_change = maximum_change.max(change.abs());
            }
        }
        let residual = sk_residual(&linear, labels, weights, inverse_rows);
        let gradient = residual.sum();
        let curvature = weights
            .iter()
            .fold(F::zero(), |accumulator, weight| accumulator + *weight)
            * inverse_rows
            * quarter;
        if curvature > F::zero() {
            let new_value =
                sk_soft_threshold(intercept - gradient / curvature, config.alpha / curvature);
            let change = new_value - intercept;
            if change != F::zero() {
                intercept = new_value;
                linear.mapv_inplace(|value| value + change);
                maximum_change = maximum_change.max(change.abs());
            }
        }
        iterations = sweep + 1;
        if maximum_change <= config.tolerance {
            return Ok((w, intercept, iterations));
        }
    }
    Err(SKError::not_converged(iterations))
}

/// The mean-scaled logistic residual `cᵢ(σ(zᵢ) − yᵢ)/n`.
fn sk_residual<F: SKFloat>(
    linear: &Array1<F>,
    labels: &[F],
    weights: &[F],
    inverse_rows: F,
) -> Array1<F> {
    Array1::from_shape_fn(linear.len(), |row| {
        weights[row] * (sk_sigmoid(linear[row]) - labels[row]) * inverse_rows
    })
}

/// The soft-threshold operator `S(z, γ) = sign(z)·max(|z| − γ, 0)`.
fn sk_soft_threshold<F: SKFloat>(value: F, threshold: F) -> F {
    if value > threshold {
        value - threshold
    } else if value < -threshold {
        value + threshold
    } else {
        F::zero()
    }
}
