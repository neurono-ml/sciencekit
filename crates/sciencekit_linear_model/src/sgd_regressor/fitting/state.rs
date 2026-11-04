//! Learning-state creation, warm-start seeding and the reported-iterate
//! refresh (spec `stochastic-gradient-regressor`, decision 13).

use ndarray::Array1;
use sciencekit_common::{SKError, SKFloat};

use super::super::coefficients::SKSGDRegressorCoefficients;
use super::super::targets::{sk_initial_slope_scale, sk_optimal_offset};
use crate::sgd::config::SKSGDConfig;
use crate::sgd::{SKLearningRateSchedule, SKLearningState, SKSGDRegressorLoss};

/// The initial learning state.
///
/// Without a warm-start source the coefficients start at zero; with one, the
/// source must carry the same feature count as the design.
pub(crate) fn sk_create_state<F: SKFloat>(
    config: &SKSGDConfig<F>,
    loss: SKSGDRegressorLoss,
    epsilon: F,
    warm_start: Option<&SKSGDRegressorCoefficients<F>>,
    features: usize,
    targets: &[F],
) -> Result<SKLearningState<F>, SKError> {
    let scale = sk_initial_slope_scale(loss, epsilon, targets);
    let schedule = SKLearningRateSchedule::new(
        config.learning_rate,
        config.initial_learning_rate,
        config.inverse_scaling_power,
        config.alpha,
        sk_optimal_offset(config.alpha, scale),
    );
    match warm_start {
        None => Ok(SKLearningState::new(
            Array1::zeros(features),
            F::zero(),
            schedule,
            config.average,
        )),
        Some(source) => {
            if source.feature_count() != features {
                return Err(SKError::ShapeMismatch {
                    expected: vec![features],
                    found: vec![source.feature_count()],
                });
            }
            Ok(SKLearningState::new(
                source.coefficients.clone(),
                source.intercept,
                schedule,
                config.average,
            ))
        }
    }
}

/// The coefficients and intercept the model reports from the current state.
pub(crate) fn sk_reported_parts<F: SKFloat>(
    state: &SKLearningState<F>,
    features: usize,
) -> (Array1<F>, F) {
    let coefficients = Array1::from_shape_fn(features, |column| state.reported_weights()[column]);
    (coefficients, state.reported_intercept())
}
