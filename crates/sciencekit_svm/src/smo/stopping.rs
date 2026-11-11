//! KKT stopping criterion, bias recovery and iteration budget bookkeeping
//! (spec `support-vector-machines`).

use sciencekit_common::{SKError, SKFloat};

use super::gradient::SKSmoState;
use super::working_set::{sk_minimum_lower_score, sk_select_first_working_index};

/// Solution returned by a converged SMO run.
#[derive(Debug, Clone, PartialEq)]
pub struct SKSmoSolution<F: SKFloat> {
    /// Optimal dual variables `alpha_t ∈ [0, U_t]`.
    pub alphas: Vec<F>,
    /// Recovered decision bias `b`.
    pub bias: F,
    /// Number of SMO iterations executed.
    pub iterations: usize,
    /// Whether the KKT stopping criterion was met within tolerance.
    pub converged: bool,
}

/// Iteration and KKT-tolerance tracker for an SMO solve.
#[derive(Debug, Clone, Copy)]
pub struct SKSmoTracker<F: SKFloat> {
    tolerance: F,
    maximum_number_of_iterations: usize,
    executed_iterations: usize,
}

impl<F: SKFloat> SKSmoTracker<F> {
    /// Create a tracker for `tolerance` and `maximum_number_of_iterations`.
    pub fn new(tolerance: F, maximum_number_of_iterations: usize) -> Self {
        Self {
            tolerance,
            maximum_number_of_iterations,
            executed_iterations: 0,
        }
    }

    /// Number of SMO iterations executed so far.
    pub fn executed_iterations(&self) -> usize {
        self.executed_iterations
    }

    /// Configured convergence tolerance.
    pub fn tolerance(&self) -> F {
        self.tolerance
    }

    /// Check KKT optimality or advance the iteration counter, returning
    /// [`SKError::NotConverged`] when the budget is exhausted.
    pub fn check_or_advance(&mut self, violation: F) -> Result<bool, SKError> {
        if violation <= self.tolerance {
            return Ok(true);
        }
        if self.executed_iterations >= self.maximum_number_of_iterations {
            return Err(SKError::NotConverged {
                iterations: self.executed_iterations,
            });
        }
        self.executed_iterations += 1;
        Ok(false)
    }
}

/// Maximal KKT violation `max_{I_up}(-y G) - min_{I_low}(-y G)` on active indices.
pub fn sk_maximal_kkt_violation<F: SKFloat>(state: &SKSmoState<F>) -> F {
    match (
        sk_select_first_working_index(state),
        sk_minimum_lower_score(state),
    ) {
        (Some((_, upper_max)), Some(lower_min)) if upper_max > lower_min => upper_max - lower_min,
        _ => F::zero(),
    }
}

/// Recover the primal bias `b` from free support vectors (or midpoint fallback).
pub fn sk_recover_smo_bias<F: SKFloat>(state: &SKSmoState<F>) -> F {
    let mut free_sum = F::zero();
    let mut free_count = 0usize;
    for index in 0..state.alphas.len() {
        if state.is_free(index) {
            free_sum = free_sum + state.negative_signed_gradient(index);
            free_count += 1;
        }
    }
    if free_count > 0 {
        return free_sum / F::from(free_count).expect("count fits scalar");
    }
    let upper_max = sk_select_first_working_index(state)
        .map(|(_, score)| score)
        .unwrap_or_else(F::zero);
    let lower_min = sk_minimum_lower_score(state).unwrap_or_else(F::zero);
    let two = F::one() + F::one();
    (upper_max + lower_min) / two
}
