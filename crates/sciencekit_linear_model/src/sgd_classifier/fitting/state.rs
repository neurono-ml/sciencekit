//! Head creation, warm-start seeding and the reported-iterate refresh (spec
//! `stochastic-gradient-classifier`, decisions 10 and 13).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKError, SKFloat, SKLabelTable};

use super::super::coefficients::SKSGDClassifierCoefficients;
use super::super::targets::{
    sk_head_targets, sk_initial_slope_scale, sk_number_of_heads, sk_optimal_offset,
    sk_positive_classes,
};
use crate::sgd::config::SKSGDConfig;
use crate::sgd::{SKLearningRateSchedule, SKLearningState, SKSGDClassifierLoss};

/// The initial learning states, one per one-vs-rest head.
///
/// Without a warm-start source every head starts at zero; with one, the source
/// must carry the same class table, feature count and head count.
pub(crate) fn sk_create_states<F: SKFloat>(
    config: &SKSGDConfig<F>,
    loss: SKSGDClassifierLoss,
    warm_start: Option<&SKSGDClassifierCoefficients<F>>,
    classes: &SKLabelTable,
    features: usize,
    class_indices: &[usize],
) -> Result<Vec<SKLearningState<F>>, SKError> {
    let number_of_classes = classes.number_of_classes();
    let number_of_heads = sk_number_of_heads(number_of_classes);
    let positive_classes = sk_positive_classes(number_of_classes);
    let scale = sk_initial_slope_scale(loss, &sk_head_targets(class_indices, positive_classes[0]));
    let schedule = SKLearningRateSchedule::new(
        config.learning_rate,
        config.initial_learning_rate,
        config.inverse_scaling_power,
        config.alpha,
        sk_optimal_offset(config.alpha, scale),
    );

    match warm_start {
        None => Ok((0..number_of_heads)
            .map(|_| {
                SKLearningState::new(
                    Array1::zeros(features),
                    F::zero(),
                    schedule.clone(),
                    config.average,
                )
            })
            .collect()),
        Some(source) => {
            sk_validate_warm_start(source, classes, features, number_of_heads)?;
            Ok((0..number_of_heads)
                .map(|head| {
                    SKLearningState::new(
                        source.coefficients.row(head).to_owned(),
                        source.intercepts[head],
                        schedule.clone(),
                        config.average,
                    )
                })
                .collect())
        }
    }
}

/// Reject a warm-start source whose shape disagrees with the current fit.
fn sk_validate_warm_start<F: SKFloat>(
    source: &SKSGDClassifierCoefficients<F>,
    classes: &SKLabelTable,
    features: usize,
    number_of_heads: usize,
) -> Result<(), SKError> {
    if source.classes != *classes {
        return Err(SKError::InvalidHyperparameter {
            name: "warm_start_from",
            reason: "the warm-start class table does not match the fitted targets".to_string(),
        });
    }
    if source.feature_count() != features {
        return Err(SKError::ShapeMismatch {
            expected: vec![features],
            found: vec![source.feature_count()],
        });
    }
    if source.coefficients.nrows() != number_of_heads {
        return Err(SKError::ShapeMismatch {
            expected: vec![number_of_heads],
            found: vec![source.coefficients.nrows()],
        });
    }
    Ok(())
}

/// The coefficients and intercepts the model reports from its current states.
pub(crate) fn sk_reported_parts<F: SKFloat>(
    states: &[SKLearningState<F>],
    features: usize,
) -> (Array2<F>, Array1<F>) {
    let coefficients = Array2::from_shape_fn((states.len(), features), |(head, column)| {
        states[head].reported_weights()[column]
    });
    let intercepts = Array1::from_shape_fn(states.len(), |head| states[head].reported_intercept());
    (coefficients, intercepts)
}
