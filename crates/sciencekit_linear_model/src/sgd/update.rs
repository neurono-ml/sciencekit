//! The single-sample stochastic-gradient update (spec
//! `stochastic-gradient-solver`, decision 3).

use sciencekit_common::SKFloat;

use super::config::SKSGDConfig;
use super::loss::SKSGDLoss;
use super::penalty::sk_apply_proximal_penalty;
use super::row::SKSGDFeatureRow;
use super::state::SKLearningState;

/// Apply one sample's update and return its weighted loss.
///
/// The gradient is `weight · slope(wᵀx + b, y) · x`; the intercept receives the
/// same learning rate and is never penalized; the weights then take the L2
/// decay and the L1 soft-threshold selected by the configured penalty.
pub(crate) fn sk_apply_single_sample_update<F: SKFloat, L: SKSGDLoss<F>, R: SKSGDFeatureRow<F>>(
    state: &mut SKLearningState<F>,
    features: R,
    target: F,
    sample_weight: F,
    loss: &L,
    config: &SKSGDConfig<F>,
) -> F {
    let prediction = features.dot_with(state.weights.view()) + state.intercept;
    let slope = loss.slope(prediction, target);
    let learning_rate = state.learning_rate();
    let weighted_slope = sample_weight * slope;

    if config.include_intercept {
        state.intercept = state.intercept - learning_rate * weighted_slope;
    }
    features.add_scaled_to(state.weights.view_mut(), -learning_rate * weighted_slope);

    sk_apply_proximal_penalty(
        state.weights.view_mut(),
        learning_rate,
        config.penalty.l1_coefficient(config.alpha, config.l1_ratio),
        config.penalty.l2_coefficient(config.alpha, config.l1_ratio),
    );

    state.step += 1;
    state.record_visited_iterate();

    sample_weight * loss.value(prediction, target)
}
