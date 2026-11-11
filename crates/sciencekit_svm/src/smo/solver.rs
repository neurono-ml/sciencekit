//! Generalized SMO quadratic-program solver (spec `support-vector-machines`).

use sciencekit_common::{SKDataView, SKError, SKFloat};
use sciencekit_math::SKKernelParameters;

use super::cache::SKKernelRowCache;
use super::gradient::SKSmoState;
use super::shrinking::{sk_reconstruct_full_gradients, sk_shrink_active_set};
use super::step::sk_clipped_pair_step;
use super::stopping::{
    SKSmoSolution, SKSmoTracker, sk_maximal_kkt_violation, sk_recover_smo_bias,
};
use super::working_set::{
    sk_curvature_floor, sk_select_first_working_index, sk_select_working_pair,
};

/// Options controlling an SMO solve.
#[derive(Debug, Clone, Copy)]
pub struct SKSmoOptions<F: SKFloat> {
    /// KKT violation stopping threshold.
    pub tolerance: F,
    /// Maximum number of SMO iterations before returning [`SKError::NotConverged`].
    pub maximum_number_of_iterations: usize,
    /// Kernel row cache budget in MiB (`0` disables caching).
    pub cache_size_mib: usize,
    /// Whether active-set shrinking is enabled.
    pub shrinking: bool,
}

fn sk_finish_solution<F: SKFloat>(state: SKSmoState<F>, iterations: usize) -> SKSmoSolution<F> {
    let bias = sk_recover_smo_bias(&state);
    SKSmoSolution {
        alphas: state.alphas,
        bias,
        iterations,
        converged: true,
    }
}

/// Solve the generalized SMO quadratic program.
pub fn sk_solve_smo_quadratic_program<F: SKFloat>(
    signs: Vec<i8>,
    upper_bounds: Vec<F>,
    linear_terms: Vec<F>,
    features: SKDataView<'_, F>,
    parameters: SKKernelParameters<F>,
    options: SKSmoOptions<F>,
) -> Result<SKSmoSolution<F>, SKError> {
    let mut cache =
        SKKernelRowCache::from_megabytes(features, parameters, options.cache_size_mib)?;
    let row_modulus = cache.diagonal().len();
    let mut state = SKSmoState::new(signs, upper_bounds, linear_terms);
    let mut tracker = SKSmoTracker::new(options.tolerance, options.maximum_number_of_iterations);
    let shrink_period = state.alphas.len().clamp(1, 1000);
    let mut has_shrunk = false;
    let two = F::one() + F::one();
    let tau = sk_curvature_floor::<F>();

    loop {
        let violation = sk_maximal_kkt_violation(&state);
        if violation <= options.tolerance {
            if has_shrunk {
                sk_reconstruct_full_gradients(&mut state, &mut cache, row_modulus)?;
                has_shrunk = false;
                if sk_maximal_kkt_violation(&state) > options.tolerance {
                    continue;
                }
            }
            return Ok(sk_finish_solution(state, tracker.executed_iterations()));
        }

        if options.shrinking
            && tracker.executed_iterations() > 0
            && tracker.executed_iterations() % shrink_period == 0
            && sk_shrink_active_set(&mut state)
        {
            has_shrunk = true;
        }

        let Some((first, first_score)) = sk_select_first_working_index(&state) else {
            if has_shrunk {
                sk_reconstruct_full_gradients(&mut state, &mut cache, row_modulus)?;
                has_shrunk = false;
                continue;
            }
            return Ok(sk_finish_solution(state, tracker.executed_iterations()));
        };

        let row_first = cache.row(first % row_modulus)?;
        let Some(pair) = sk_select_working_pair(
            &state,
            first,
            first_score,
            cache.diagonal(),
            &row_first,
            row_modulus,
        ) else {
            if has_shrunk {
                sk_reconstruct_full_gradients(&mut state, &mut cache, row_modulus)?;
                has_shrunk = false;
                continue;
            }
            return Ok(sk_finish_solution(state, tracker.executed_iterations()));
        };

        tracker.check_or_advance(pair.violation)?;

        let row_second = cache.row(pair.second % row_modulus)?;
        let diag = cache.diagonal();
        let raw_curvature = diag[pair.first % row_modulus] + diag[pair.second % row_modulus]
            - two * row_first[pair.second % row_modulus];
        let curvature = if raw_curvature > F::zero() {
            raw_curvature
        } else {
            tau
        };
        let (new_first, new_second) =
            sk_clipped_pair_step(&state, pair.first, pair.second, curvature);
        if new_first == state.alphas[pair.first] && new_second == state.alphas[pair.second] {
            if has_shrunk {
                sk_reconstruct_full_gradients(&mut state, &mut cache, row_modulus)?;
                has_shrunk = false;
                continue;
            }
            return Ok(sk_finish_solution(state, tracker.executed_iterations()));
        }
        state.update_pair(
            pair.first,
            pair.second,
            new_first,
            new_second,
            &row_first,
            &row_second,
            row_modulus,
        );
    }
}
