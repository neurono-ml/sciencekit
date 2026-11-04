//! Companion tests for the five classifier losses (spec
//! `stochastic-gradient-classifier`).

use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKClassifierPredictor, SKError, SKSupervisedFit};

use super::SKSGDClassifier;
use crate::sgd::SKConvergencePolicy;
use crate::sgd::SKSGDClassifierLoss;
use crate::test_datasets::{labels_i64, separable_binary_design};

fn assert_exact(loss: SKSGDClassifierLoss) {
    let (features, raw) = separable_binary_design();
    let labels = labels_i64(&raw);
    let model = SKSGDClassifier::<f64>::build()
        .loss(loss)
        .shuffle(false)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let predicted = model.predict_labels(features.view()).unwrap();
    assert_eq!(
        predicted.to_vec(),
        labels.to_vec(),
        "the {loss:?} fit was not exact"
    );
}

#[test]
fn hinge_and_perceptron_are_exact_on_a_separable_design() {
    assert_exact(SKSGDClassifierLoss::Hinge);
    assert_exact(SKSGDClassifierLoss::Perceptron);
}

#[test]
fn squared_hinge_and_modified_huber_converge() {
    let (features, raw) = separable_binary_design();
    let labels = labels_i64(&raw);
    for loss in [
        SKSGDClassifierLoss::SquaredHinge,
        SKSGDClassifierLoss::ModifiedHuber,
    ] {
        let model = SKSGDClassifier::<f64>::build()
            .loss(loss)
            .shuffle(false)
            .tolerance(1e-2)
            .build()
            .unwrap()
            .fit(features.view(), labels.view())
            .unwrap();
        assert!(model.converged(), "the {loss:?} fit did not converge");
        assert!(model.epochs_taken() > 0);
    }
}

#[test]
fn log_loss_is_calibrated_and_normalized() {
    let (features, raw) = separable_binary_design();
    let labels = labels_i64(&raw);
    let model = SKSGDClassifier::<f64>::build()
        .loss(SKSGDClassifierLoss::LogLoss)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let probabilities = model.predict_probabilities(features.view()).unwrap();
    assert_eq!(probabilities.dim(), (6, 2));
    for row in probabilities.rows() {
        assert!((row.sum() - 1.0).abs() < 1e-9);
    }
    let predicted = model.predict_labels(features.view()).unwrap();
    for row in 0..probabilities.nrows() {
        let best = if probabilities[[row, 1]] > probabilities[[row, 0]] {
            1
        } else {
            0
        };
        assert_eq!(predicted[row], best);
    }
}

#[test]
fn probabilities_are_refused_for_non_probabilistic_losses() {
    let (features, raw) = separable_binary_design();
    let labels = labels_i64(&raw);
    for loss in [
        SKSGDClassifierLoss::Hinge,
        SKSGDClassifierLoss::SquaredHinge,
        SKSGDClassifierLoss::Perceptron,
    ] {
        let model = SKSGDClassifier::<f64>::build()
            .loss(loss)
            .convergence_policy(SKConvergencePolicy::Permissive)
            .build()
            .unwrap()
            .fit(features.view(), labels.view())
            .unwrap();
        let error = model.predict_probabilities(features.view()).unwrap_err();
        assert!(
            matches!(
                error,
                SKError::UnsupportedOperation { operation, .. }
                    if operation == "predict_probabilities"
            ),
            "unexpected error for {loss:?}"
        );
    }
}
