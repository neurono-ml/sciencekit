//! The LIBLINEAR L2-regularized dual coordinate descent for a binary head
//! (spec `logistic-solver`, Decision 3).
//!
//! The dual is the box-constrained problem over `α ∈ [0, gᵢ]` with the scheme
//! `w = (1/α)·Σ αᵢ yᵢ x̃ᵢ`; each coordinate is updated by a Newton step on the
//! scalar subproblem. As in scikit-learn's LIBLINEAR path, the intercept is
//! augmented as a constant feature and is therefore regularised.

use ndarray::Array1;
use sciencekit_common::{SKError, SKFloat};

use crate::logistic::config::SKLogisticConfig;
use crate::logistic::design::SKLogisticDesign;

/// Solve a binary head with the LIBLINEAR L2 dual coordinate descent.
pub(crate) fn sk_solve_binary_head_liblinear_l2<F, D>(
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
    let half = F::from(0.5).expect("fits");
    let signed: Vec<F> = labels
        .iter()
        .map(|label| if *label > half { F::one() } else { -F::one() })
        .collect();
    let bounds: Vec<F> = weights
        .iter()
        .map(|weight| *weight * inverse_rows)
        .collect();
    let intercept_term = if config.include_intercept {
        F::one()
    } else {
        F::zero()
    };
    let mut alphas: Vec<F> = bounds.iter().map(|bound| *bound * half).collect();
    let mut w = Array1::<F>::zeros(columns);
    let mut intercept = F::zero();
    for row in 0..rows {
        let scale = alphas[row] * signed[row] / config.alpha;
        design.add_row_scaled(row, scale, w.view_mut());
        intercept = intercept + scale;
    }

    let mut iterations = 0usize;
    for sweep in 0..config.maximum_number_of_iterations {
        let mut maximum_change = F::zero();
        for row in 0..rows {
            if bounds[row] <= F::zero() {
                continue;
            }
            let score = design.row_dot(row, w.view()) + intercept;
            let dual_gradient =
                signed[row] * score + (alphas[row] / (bounds[row] - alphas[row])).ln();
            let curvature = (design.row_squared_norm(row) + intercept_term) / config.alpha;
            let change = sk_dual_newton(alphas[row], bounds[row], dual_gradient, curvature);
            if change == F::zero() {
                continue;
            }
            let scale = change * signed[row] / config.alpha;
            design.add_row_scaled(row, scale, w.view_mut());
            intercept = intercept + scale;
            alphas[row] = alphas[row] + change;
            maximum_change = maximum_change.max(change.abs());
        }
        iterations = sweep + 1;
        if maximum_change <= config.tolerance {
            return Ok((w, intercept, iterations));
        }
    }
    Err(SKError::not_converged(iterations))
}

/// Solve `H(d) = G + q·d + log((a+d)/(C−a−d)) − log(a/(C−a)) = 0` for the
/// bounded coordinate step `d`, via Newton iterations clamped to the box.
fn sk_dual_newton<F: SKFloat>(alpha: F, bound: F, gradient: F, curvature: F) -> F {
    let epsilon = bound * F::from(1e-10).expect("fits");
    let lower = -alpha + epsilon;
    let upper = bound - alpha - epsilon;
    let base = (alpha / (bound - alpha)).ln();
    let mut change = F::zero();
    for _ in 0..50 {
        let current = alpha + change;
        let value = gradient + curvature * change + (current / (bound - current)).ln() - base;
        if value.abs() <= F::from(1e-12).expect("fits") {
            break;
        }
        let derivative = curvature + F::one() / current + F::one() / (bound - current);
        let mut next = change - value / derivative;
        if next <= lower {
            next = lower;
        } else if next >= upper {
            next = upper;
        }
        if (next - change).abs() <= F::from(1e-15).expect("fits") {
            change = next;
            break;
        }
        change = next;
    }
    change
}
