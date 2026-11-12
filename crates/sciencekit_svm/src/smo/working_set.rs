//! Second-order working-set selection (Fan, Chen & Lin 2005)
//! (spec `support-vector-machines`).

use sciencekit_common::SKFloat;

use super::gradient::SKSmoState;

/// Curvature floor guarding non-PSD kernels and duplicate rows.
pub(crate) fn sk_curvature_floor<F: SKFloat>() -> F {
    F::from(1e-12).expect("1e-12 fits SKFloat")
}

/// Selected `(first, second)` working pair and its KKT diagnostics.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SKSmoWorkingPair<F: SKFloat> {
    /// Index `i ∈ I_up` maximizing `-y_i G_i`.
    pub first: usize,
    /// Index `j ∈ I_low` maximizing the second-order objective gain.
    pub second: usize,
    /// Maximal KKT violation `max_{I_up}(-y G) - min_{I_low}(-y G)`.
    pub violation: F,
    /// Unclipped second-order objective gain `b_ij² / (2 a_ij)`.
    pub second_order_gain: F,
}

/// Select `first ∈ I_up(alpha)` maximizing `-y_t G_t` (smallest index breaks ties).
pub fn sk_select_first_working_index<F: SKFloat>(state: &SKSmoState<F>) -> Option<(usize, F)> {
    let mut best: Option<(usize, F)> = None;
    for &index in &state.active_indices {
        if !state.is_upper(index) {
            continue;
        }
        let score = state.negative_signed_gradient(index);
        match best {
            None => best = Some((index, score)),
            Some((_, best_score)) if score > best_score => best = Some((index, score)),
            _ => {}
        }
    }
    best
}

/// Minimum of `-y_t G_t` across active indices in `I_low(alpha)`.
pub fn sk_minimum_lower_score<F: SKFloat>(state: &SKSmoState<F>) -> Option<F> {
    let mut minimum: Option<F> = None;
    for &index in &state.active_indices {
        if !state.is_lower(index) {
            continue;
        }
        let score = state.negative_signed_gradient(index);
        match minimum {
            None => minimum = Some(score),
            Some(current) if score < current => minimum = Some(score),
            _ => {}
        }
    }
    minimum
}

/// Select `second ∈ I_low(alpha)` maximizing the second-order gain paired with `first`.
pub fn sk_select_working_pair<F: SKFloat>(
    state: &SKSmoState<F>,
    first: usize,
    first_score: F,
    kernel_diagonal: &[F],
    first_kernel_row: &[F],
    row_modulus: usize,
) -> Option<SKSmoWorkingPair<F>> {
    let minimum_lower = sk_minimum_lower_score(state)?;
    let violation = first_score - minimum_lower;
    let two = F::one() + F::one();
    let tau = sk_curvature_floor::<F>();
    let first_row = first % row_modulus;
    let diagonal_first = kernel_diagonal[first_row];

    let mut best_second: Option<(usize, F)> = None;
    for &candidate in &state.active_indices {
        if candidate == first || !state.is_lower(candidate) {
            continue;
        }
        let linear_gap = first_score - state.negative_signed_gradient(candidate);
        if linear_gap <= F::zero() {
            continue;
        }
        let candidate_row = candidate % row_modulus;
        let raw_curvature =
            diagonal_first + kernel_diagonal[candidate_row] - two * first_kernel_row[candidate_row];
        let curvature = if raw_curvature > F::zero() {
            raw_curvature
        } else {
            tau
        };
        let gain = (linear_gap * linear_gap) / (two * curvature);
        match best_second {
            None => best_second = Some((candidate, gain)),
            Some((_, best_gain)) if gain > best_gain => best_second = Some((candidate, gain)),
            _ => {}
        }
    }

    best_second.map(|(second, second_order_gain)| SKSmoWorkingPair {
        first,
        second,
        violation,
        second_order_gain,
    })
}
