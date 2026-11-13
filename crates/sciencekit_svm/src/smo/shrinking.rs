//! Active-set shrinking and full-gradient reconstruction for SMO
//! (spec `support-vector-machines`).

use sciencekit_common::{SKError, SKFloat};

use super::cache::SKKernelRowCache;
use super::gradient::SKSmoState;
use super::working_set::{sk_minimum_lower_score, sk_select_first_working_index};

/// Shrink active variables that sit strictly at a bound with multipliers
/// pointing into that bound. Keeps at least two active indices.
pub fn sk_shrink_active_set<F: SKFloat>(state: &mut SKSmoState<F>) -> bool {
    let Some((_, upper_max)) = sk_select_first_working_index(state) else {
        return false;
    };
    let Some(lower_min) = sk_minimum_lower_score(state) else {
        return false;
    };
    let mut retained = Vec::with_capacity(state.active_indices.len());
    for &index in &state.active_indices {
        if state.is_free(index) {
            retained.push(index);
            continue;
        }
        let score = state.negative_signed_gradient(index);
        let shrinkable = match (state.is_upper(index), state.is_lower(index)) {
            (false, false) => true,
            (true, false) => score < lower_min,
            (false, true) => score > upper_max,
            (true, true) => false,
        };
        if !shrinkable {
            retained.push(index);
        }
    }
    if retained.len() >= 2 && retained.len() < state.active_indices.len() {
        state.active_indices = retained;
        return true;
    }
    false
}

/// Restore all indices to the active set and recompute exact gradients over
/// every variable from non-zero `alpha` entries.
pub fn sk_reconstruct_full_gradients<F: SKFloat>(
    state: &mut SKSmoState<F>,
    cache: &mut SKKernelRowCache<'_, F>,
    row_modulus: usize,
) -> Result<(), SKError> {
    let count = state.alphas.len();
    state.active_indices = (0..count).collect();
    state.gradients.clone_from(&state.linear_terms);

    for (source, alpha) in state.alphas.iter().copied().enumerate() {
        if alpha == F::zero() {
            continue;
        }
        let signed_alpha = state.sign_scalar(source) * alpha;
        let kernel_row = cache.row(source % row_modulus)?;
        state
            .gradients
            .iter_mut()
            .zip(state.signs.iter())
            .enumerate()
            .for_each(|(target, (gradient, &sign))| {
                let increment = signed_alpha * kernel_row[target % row_modulus];
                if sign > 0 {
                    *gradient = *gradient + increment;
                } else {
                    *gradient = *gradient - increment;
                }
            });
    }
    Ok(())
}
