//! The inner sweeping iteration: soft-threshold coordinate updates, the
//! unpenalized residual-mean intercept and the convergence test (spec
//! `coordinate-descent-solver`).

use ndarray::{Array1, azip};
use sciencekit_common::SKFloat;

use super::super::access::SKColumnAccess;
use super::super::config::{SKCoordinateDescentConfig, SKSelection};

/// Run sweeps until convergence or the budget is exhausted; returns the sweeps
/// spent and whether the maximum coordinate change fell to the tolerance.
#[allow(clippy::too_many_arguments)]
pub(super) fn sk_run_sweeps<F: SKFloat, A: SKColumnAccess<F>>(
    access: &A,
    residual: &mut Array1<F>,
    coefficients: &mut Array1<F>,
    intercept: &mut F,
    active: &[bool],
    denominators: &[F],
    l1: F,
    l2: F,
    inv_n: F,
    include_intercept: bool,
    config: &SKCoordinateDescentConfig<F>,
    budget: usize,
    seed: &mut u64,
) -> (usize, bool) {
    let columns = access.columns_count();
    for iteration in 1..=budget {
        let order = match config.selection {
            SKSelection::Cyclic => (0..columns).collect::<Vec<usize>>(),
            SKSelection::Random => sk_next_permutation(columns, seed),
        };
        let mut maximum_delta = F::zero();
        for column in order {
            if !active[column] {
                continue;
            }
            let column_scale = denominators[column] - l2;
            let gradient = access.signed_column_dot(column, residual.view()) * inv_n;
            let partial = gradient + column_scale * coefficients[column];
            let thresholded = sk_soft_threshold(partial, l1);
            let mut new_value = thresholded / denominators[column];
            if config.positive && new_value < F::zero() {
                new_value = F::zero();
            }
            let delta = new_value - coefficients[column];
            if delta != F::zero() {
                access.update_residual(column, delta, residual.view_mut());
                coefficients[column] = new_value;
                if delta.abs() > maximum_delta {
                    maximum_delta = delta.abs();
                }
            }
        }
        if include_intercept {
            let shift = residual.sum() * inv_n;
            if shift != F::zero() {
                azip!((entry in &mut *residual) { *entry = *entry - shift; });
                *intercept = *intercept + shift;
                if shift.abs() > maximum_delta {
                    maximum_delta = shift.abs();
                }
            }
        }
        if maximum_delta <= config.tolerance {
            return (iteration, true);
        }
    }
    (budget, false)
}

/// The soft-threshold operator `S(z, γ) = sign(z)·max(|z| − γ, 0)`.
pub(super) fn sk_soft_threshold<F: SKFloat>(value: F, threshold: F) -> F {
    if value > threshold {
        value - threshold
    } else if value < -threshold {
        value + threshold
    } else {
        F::zero()
    }
}

/// A deterministic pseudo-random permutation of `0..columns` (Fisher–Yates over
/// an xorshift64* stream), so the random selection is reproducible.
fn sk_next_permutation(columns: usize, seed: &mut u64) -> Vec<usize> {
    let mut order: Vec<usize> = (0..columns).collect();
    for index in (1..columns).rev() {
        *seed ^= *seed << 13;
        *seed ^= *seed >> 7;
        *seed ^= *seed << 17;
        let swap_with = (*seed as usize) % (index + 1);
        order.swap(index, swap_with);
    }
    order
}
