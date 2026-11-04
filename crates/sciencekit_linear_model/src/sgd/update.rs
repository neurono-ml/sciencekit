//! The single-sample stochastic-gradient update (spec
//! `stochastic-gradient-solver`, decision 3).

use ndarray::{ArrayView1, azip};
use sciencekit_common::SKFloat;

use super::config::SKSGDConfig;
use super::loss::SKSGDLoss;
use super::penalty::sk_apply_proximal_penalty;
use super::state::SKLearningState;

/// Apply one sample's update and return its weighted loss.
///
/// The gradient is `weight · slope(wᵀx + b, y) · x`; the intercept receives the
/// same learning rate and is never penalized; the weights then take the L2
/// decay and the L1 soft-threshold selected by the configured penalty.
pub(crate) fn sk_apply_single_sample_update<F: SKFloat, L: SKSGDLoss<F>>(
    state: &mut SKLearningState<F>,
    features: ArrayView1<F>,
    target: F,
    sample_weight: F,
    loss: &L,
    config: &SKSGDConfig<F>,
) -> F {
    let prediction = state.predicted(features);
    let slope = loss.slope(prediction, target);
    let learning_rate = state.learning_rate();
    let weighted_slope = sample_weight * slope;

    if config.include_intercept {
        state.intercept = state.intercept - learning_rate * weighted_slope;
    }
    azip!((weight in &mut state.weights, &feature in &features) {
        *weight = *weight - learning_rate * weighted_slope * feature;
    });

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
