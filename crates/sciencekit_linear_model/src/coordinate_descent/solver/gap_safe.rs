//! The gap-safe screening applied during the outer loop: deactivate active
//! coordinates that are exactly zero and provably zero (spec
//! `coordinate-descent-solver`, Decision 5).

use ndarray::{Array1, ArrayView1};
use sciencekit_common::SKFloat;

use super::super::access::SKColumnAccess;
use super::super::screening::{sk_dual_feasible_scale, sk_gap_safe_radius, sk_is_safely_zero};
use super::orchestration::sk_full_gradient;

/// Deactivate active coordinates that are exactly zero and provably zero by the
/// gap-safe test, using the current solution's dual-feasible point and radius.
#[allow(clippy::too_many_arguments)]
pub(super) fn sk_apply_gap_safe_screening<F: SKFloat, A: SKColumnAccess<F>>(
    access: &A,
    residual: &Array1<F>,
    target: &ArrayView1<F>,
    coefficients: &Array1<F>,
    active: &mut [bool],
    l1: F,
    l2: F,
    rows: usize,
) {
    if l1 <= F::zero() || rows == 0 {
        return;
    }
    // The gap-safe radius below is derived for the pure-L1 (lasso) objective.
    // The elastic net still uses the strong initial set plus the KKT
    // reactivation guarantee, but not this gap-based elimination (design
    // Decision 5).
    if l2 > F::zero() {
        return;
    }
    let row_count = F::from(rows).expect("row counts fit in the scalar");
    let inv_n = F::one() / row_count;
    let gradient = sk_full_gradient(access, residual.view(), inv_n);
    let infinity_norm = gradient
        .iter()
        .fold(F::zero(), |best, value| best.max(value.abs()));
    let scale = sk_dual_feasible_scale(infinity_norm, l1);
    let duality_gap = sk_duality_gap(
        residual,
        target,
        coefficients,
        l1,
        l2,
        row_count,
        scale,
        inv_n,
    );
    let radius = sk_gap_safe_radius(duality_gap, rows);
    let mut deactivate = Vec::new();
    for column in 0..access.columns_count() {
        if !active[column] || coefficients[column] != F::zero() {
            continue;
        }
        let dual_dot = (gradient[column] / scale).abs();
        let column_norm = access.column_squared_norm(column).sqrt();
        if sk_is_safely_zero(dual_dot, column_norm, radius, l1) {
            deactivate.push(column);
        }
    }
    deactivate
        .into_iter()
        .for_each(|column| active[column] = false);
}

/// The primal–dual gap at the current solution and dual-feasible point
/// `z = −r / (n · scale)`.
#[allow(clippy::too_many_arguments)]
fn sk_duality_gap<F: SKFloat>(
    residual: &Array1<F>,
    target: &ArrayView1<F>,
    coefficients: &Array1<F>,
    l1: F,
    l2: F,
    row_count: F,
    scale: F,
    inv_n: F,
) -> F {
    let two = F::one() + F::one();
    let residual_squared = residual.dot(residual);
    let l1_sum = coefficients
        .iter()
        .map(|value| value.abs())
        .fold(F::zero(), |accumulator, term| accumulator + term);
    let l2_sum = coefficients
        .iter()
        .map(|value| *value * *value)
        .fold(F::zero(), |accumulator, term| accumulator + term);
    let primal = residual_squared * inv_n / two + l1 * l1_sum + l2 * l2_sum / two;
    let z = residual.mapv(|value| -value * inv_n / scale);
    let z_dot_target = z.dot(target);
    let z_squared = z.dot(&z);
    let dual = -z_dot_target - row_count * z_squared / two;
    primal - dual
}
