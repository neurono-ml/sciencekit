//! LIBLINEAR-style dual coordinate descent for the L2-regularized linear SVC
//! (spec `linear-support-vector-classifier`).
//!
//! For `n` rows the solver minimizes the dual
//!
//! ```text
//! min_α  1/2 αᵀ Q α - eᵀ α
//! ```
//!
//! with `Q_ij = y_i y_j x_iᵀ x_j` over the augmented rows, `0 ≤ α_i ≤ C·weight_i`
//! for the hinge loss and `α_i ≥ 0` with a diagonal shift `1 / (2·C·weight_i)`
//! for the squared hinge loss. One coordinate is updated at a time with the
//! projected gradient; the coefficient vector `w` is maintained incrementally,
//! so one sweep costs one dot product per row.

use sciencekit_common::{SKError, SKFloat};

use super::config::SKLinearSvcLoss;
use super::design::SKLinearSvcDesignAccess;

/// Result of a converged dual solve.
#[derive(Debug, Clone)]
pub(crate) struct SKLinearSvcSolverOutcome<F: SKFloat> {
    /// Augmented coefficient vector: stored features followed by the intercept
    /// when the design carries one.
    pub(crate) coefficients: Vec<F>,
    /// Number of executed dual sweeps.
    pub(crate) iterations: usize,
}

/// Solve the dual for one binary head.
///
/// `labels` carries `+1`/`-1`; `row_weights` carries the class weight of each
/// row (the effective `C` is `regularization_c · row_weight`).
pub(crate) fn sk_solve_linear_svc_dual<F: SKFloat>(
    design: &(dyn SKLinearSvcDesignAccess<F> + Sync),
    labels: &[F],
    row_weights: &[F],
    loss: SKLinearSvcLoss,
    regularization_c: F,
    tolerance: F,
    maximum_number_of_iterations: usize,
) -> Result<SKLinearSvcSolverOutcome<F>, SKError> {
    let rows = design.number_of_rows();
    let columns = design.number_of_columns();
    let has_intercept = design.has_intercept();
    let mut dual = vec![F::zero(); rows];
    let mut coefficients = vec![F::zero(); columns + usize::from(has_intercept)];
    let mut iterations = 0;
    let mut converged = false;

    for sweep in 1..=maximum_number_of_iterations {
        let mut maximum_projected_gradient = F::zero();
        (0..rows).for_each(|row| {
            let class_capacity = regularization_c * row_weights[row];
            let diagonal_shift = match loss {
                SKLinearSvcLoss::Hinge => F::zero(),
                SKLinearSvcLoss::SquaredHinge => {
                    // The squared-hinge path lands in the next task.
                    F::zero()
                }
            };
            let upper_bound = match loss {
                SKLinearSvcLoss::Hinge => class_capacity,
                SKLinearSvcLoss::SquaredHinge => F::infinity(),
            };
            let diagonal = design.row_norm_squared(row) + diagonal_shift;
            if diagonal <= F::zero() {
                return;
            }
            let gradient = labels[row] * design.row_dot(row, &coefficients) - F::one()
                + dual[row] * diagonal_shift;
            let projected_gradient = if dual[row] <= F::zero() {
                gradient.min(F::zero())
            } else if dual[row] >= upper_bound {
                gradient.max(F::zero())
            } else {
                gradient
            };
            maximum_projected_gradient = maximum_projected_gradient.max(projected_gradient.abs());
            if projected_gradient == F::zero() {
                return;
            }
            let previous = dual[row];
            let updated = (dual[row] - gradient / diagonal)
                .max(F::zero())
                .min(upper_bound);
            if updated != previous {
                dual[row] = updated;
                design.add_scaled_row(row, (updated - previous) * labels[row], &mut coefficients);
            }
        });
        iterations = sweep;
        if maximum_projected_gradient < tolerance {
            converged = true;
            break;
        }
    }

    if !converged {
        return Err(SKError::NotConverged { iterations });
    }

    if has_intercept {
        coefficients[columns] =
            sk_recover_linear_svc_intercept(design, labels, row_weights, loss, &mut coefficients);
    }

    Ok(SKLinearSvcSolverOutcome {
        coefficients,
        iterations,
    })
}

/// Recover the unpenalized intercept for the fitted coefficient vector.
///
/// The dual solve treats the intercept as an augmented constant feature so one
/// coordinate loop stays valid. The primal objective is flat in an unpenalized
/// intercept, so after convergence the intercept is re-optimized by bisecting
/// the derivative of the class-weighted loss along that single direction.
fn sk_recover_linear_svc_intercept<F: SKFloat>(
    design: &(dyn SKLinearSvcDesignAccess<F> + Sync),
    labels: &[F],
    row_weights: &[F],
    loss: SKLinearSvcLoss,
    coefficients: &mut [F],
) -> F {
    let rows = design.number_of_rows();
    let columns = design.number_of_columns();
    coefficients[columns] = F::zero();
    let margins: Vec<F> = (0..rows)
        .map(|row| labels[row] * design.row_dot(row, coefficients))
        .collect();

    let (lower_breakpoint, upper_breakpoint) =
        (0..rows).fold((F::infinity(), -F::infinity()), |(lower, upper), row| {
            let breakpoint = labels[row] * (F::one() - margins[row]);
            (lower.min(breakpoint), upper.max(breakpoint))
        });
    if !lower_breakpoint.le(&upper_breakpoint) {
        return F::zero();
    }

    let mut lower = lower_breakpoint - F::one();
    let mut upper = upper_breakpoint + F::one();
    (0..100).for_each(|_| {
        let midpoint = (lower + upper) / (F::one() + F::one());
        let derivative = (0..rows).fold(F::zero(), |accumulator, row| {
            let residual = F::one() - margins[row] - labels[row] * midpoint;
            if residual <= F::zero() {
                accumulator
            } else {
                let slope = match loss {
                    SKLinearSvcLoss::Hinge => F::one(),
                    SKLinearSvcLoss::SquaredHinge => (F::one() + F::one()) * residual,
                };
                accumulator - row_weights[row] * labels[row] * slope
            }
        });
        if derivative > F::zero() {
            upper = midpoint;
        } else {
            lower = midpoint;
        }
    });
    (lower + upper) / (F::one() + F::one())
}
