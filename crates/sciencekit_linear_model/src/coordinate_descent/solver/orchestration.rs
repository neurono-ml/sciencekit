//! The single-alpha screened outer loop: the initial active set, the
//! convergence/reactivation loop and the full-gradient helper (spec
//! `coordinate-descent-solver`).

use ndarray::{Array1, ArrayView1};
use sciencekit_common::{SKError, SKFloat};

use super::super::access::SKColumnAccess;
use super::super::config::SKCoordinateDescentConfig;
use super::super::screening::sk_kkt_violations;
use super::super::solution::SKCoordinateDescentAlphaSolution;
use super::gap_safe::sk_apply_gap_safe_screening;
use super::iteration::sk_run_sweeps;

/// Fixed seed for the deterministic random sweep order.
const RANDOM_SEED: u64 = 0x9E37_79B9_7F4A_7C15;

/// Solve one `alpha`, screening the initial active set when enabled.
pub(crate) fn sk_solve_alpha<F: SKFloat, A: SKColumnAccess<F>>(
    access: &A,
    target: ArrayView1<F>,
    alpha: F,
    l1_ratio: F,
    include_intercept: bool,
    config: &SKCoordinateDescentConfig<F>,
) -> Result<SKCoordinateDescentAlphaSolution<F>, SKError> {
    let active = sk_initial_active_set(access, target, alpha * l1_ratio, include_intercept, config);
    sk_solve_alpha_with_initial_active(
        access,
        target,
        alpha,
        l1_ratio,
        include_intercept,
        config,
        active,
    )
}

/// Solve one `alpha` on a caller-provided initial active set, reactivating any
/// coordinate the KKT check finds violating (the correctness guarantee of
/// screening).
pub(crate) fn sk_solve_alpha_with_initial_active<F: SKFloat, A: SKColumnAccess<F>>(
    access: &A,
    target: ArrayView1<F>,
    alpha: F,
    l1_ratio: F,
    include_intercept: bool,
    config: &SKCoordinateDescentConfig<F>,
    mut active: Vec<bool>,
) -> Result<SKCoordinateDescentAlphaSolution<F>, SKError> {
    let columns = access.columns_count();
    let rows = access.rows_count();
    let n = F::from(rows).expect("row counts fit in the scalar");
    let inv_n = F::one() / n;
    let l1 = alpha * l1_ratio;
    let l2 = alpha * (F::one() - l1_ratio);
    let target_mean = target.sum() * inv_n;
    let mut intercept = if include_intercept {
        target_mean
    } else {
        F::zero()
    };
    let mut residual = if include_intercept {
        target.mapv(|value| value - target_mean)
    } else {
        target.to_owned()
    };
    let mut coefficients = Array1::<F>::zeros(columns);
    let denominators: Vec<F> = (0..columns)
        .map(|column| access.column_squared_norm(column) * inv_n + l2)
        .collect();

    let mut total_iterations = 0usize;
    let mut seed = RANDOM_SEED;
    loop {
        let budget = config
            .maximum_number_of_iterations
            .saturating_sub(total_iterations);
        if budget == 0 {
            return Err(SKError::not_converged(total_iterations));
        }
        let (sweeps, converged) = sk_run_sweeps(
            access,
            &mut residual,
            &mut coefficients,
            &mut intercept,
            &active,
            &denominators,
            l1,
            l2,
            inv_n,
            include_intercept,
            config,
            budget,
            &mut seed,
        );
        total_iterations += sweeps;
        if !converged {
            return Err(SKError::not_converged(total_iterations));
        }
        if !config.screening {
            break;
        }
        sk_apply_gap_safe_screening(
            access,
            &residual,
            &target,
            &coefficients,
            &mut active,
            l1,
            l2,
            rows,
        );
        let gradient = sk_full_gradient(access, residual.view(), inv_n);
        let violators = sk_kkt_violations(&gradient, &active, l1, config.tolerance);
        if violators.is_empty() {
            break;
        }
        violators.into_iter().for_each(|index| active[index] = true);
    }
    let screened_count = active.iter().filter(|is_active| !**is_active).count();
    Ok(SKCoordinateDescentAlphaSolution {
        coefficients,
        intercept,
        iterations: total_iterations,
        screened_count,
    })
}

/// The initial active set: every coordinate when screening is disabled, and the
/// strong set (gradient at the origin exceeds the L1 threshold) otherwise.
fn sk_initial_active_set<F: SKFloat, A: SKColumnAccess<F>>(
    access: &A,
    target: ArrayView1<F>,
    l1: F,
    include_intercept: bool,
    config: &SKCoordinateDescentConfig<F>,
) -> Vec<bool> {
    let columns = access.columns_count();
    if !config.screening {
        return vec![true; columns];
    }
    let rows = access.rows_count();
    let inv_n = F::one() / F::from(rows).expect("row counts fit in the scalar");
    let residual = if include_intercept {
        let mean = target.sum() * inv_n;
        target.mapv(|value| value - mean)
    } else {
        target.to_owned()
    };
    (0..columns)
        .map(|column| access.signed_column_dot(column, residual.view()) * inv_n)
        .map(|gradient| gradient.abs() > l1)
        .collect()
}

/// The full gradient `xⱼᵀr / n` at the current residual.
pub(super) fn sk_full_gradient<F: SKFloat, A: SKColumnAccess<F>>(
    access: &A,
    residual: ArrayView1<F>,
    inv_n: F,
) -> Vec<F> {
    (0..access.columns_count())
        .map(|column| access.signed_column_dot(column, residual) * inv_n)
        .collect()
}
