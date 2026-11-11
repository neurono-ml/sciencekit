//! Box-clipped two-variable SMO step (spec `support-vector-machines`).

use sciencekit_common::SKFloat;

use super::gradient::SKSmoState;

/// Compute the box-clipped two-variable step `(new_first, new_second)` keeping
/// `y_i * alpha_i + y_j * alpha_j` constant.
pub(crate) fn sk_clipped_pair_step<F: SKFloat>(
    state: &SKSmoState<F>,
    first: usize,
    second: usize,
    curvature: F,
) -> (F, F) {
    let linear_gap =
        state.negative_signed_gradient(first) - state.negative_signed_gradient(second);
    let raw_delta = (linear_gap / curvature).max(F::zero());
    let alpha_first = state.alphas[first];
    let alpha_second = state.alphas[second];
    let upper_first = state.upper_bounds[first];
    let upper_second = state.upper_bounds[second];

    let max_delta_first = if state.signs[first] > 0 {
        upper_first - alpha_first
    } else {
        alpha_first
    };
    let max_delta_second = if state.signs[second] > 0 {
        alpha_second
    } else {
        upper_second - alpha_second
    };

    let delta = raw_delta.min(max_delta_first).min(max_delta_second);
    let new_first = if delta >= max_delta_first {
        if state.signs[first] > 0 {
            upper_first
        } else {
            F::zero()
        }
    } else {
        (alpha_first + state.sign_scalar(first) * delta)
            .max(F::zero())
            .min(upper_first)
    };

    let new_second = if delta >= max_delta_second {
        if state.signs[second] > 0 {
            F::zero()
        } else {
            upper_second
        }
    } else {
        (alpha_second - state.sign_scalar(second) * delta)
            .max(F::zero())
            .min(upper_second)
    };

    (new_first, new_second)
}
