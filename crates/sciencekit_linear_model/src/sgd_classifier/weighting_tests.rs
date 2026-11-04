//! Companion tests for class weighting in the SGD classifier (spec
//! `stochastic-gradient-classifier`).

use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKClassifierPredictor, SKError, SKSupervisedFit};

use super::SKSGDClassifier;
use crate::class_weight::SKClassWeight;
use crate::sgd::SKConvergencePolicy;
use crate::test_datasets::{assert_close, imbalanced_binary_design, labels_i64};

#[test]
fn balanced_improves_the_minority_class() {
    let (features, raw) = imbalanced_binary_design();
    let labels = labels_i64(&raw);
    let unbalanced = SKSGDClassifier::<f64>::build()
        .shuffle(false)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .maximum_number_of_iterations(200)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let balanced = SKSGDClassifier::<f64>::build()
        .class_weight(SKClassWeight::Balanced)
        .shuffle(false)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .maximum_number_of_iterations(200)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let minority = |model: &super::SKSGDClassifierModel<f64>| {
        model
            .predict_labels(features.view())
            .unwrap()
            .iter()
            .filter(|&&label| label == 1)
            .count()
    };
    assert!(
        minority(&balanced) >= minority(&unbalanced),
        "balanced weighting did not improve minority recall"
    );
}

#[test]
fn explicit_weights_are_applied_in_class_order() {
    let (features, raw) = imbalanced_binary_design();
    let labels = labels_i64(&raw);
    // Nine samples of class 0 and two of class 1: the balanced weights are
    // 11 / (2 · 9) and 11 / (2 · 2), in that order.
    let balanced_weight = 11.0 / 18.0;
    let minority_weight = 2.75;
    let fit = |class_weight: SKClassWeight<f64>| {
        SKSGDClassifier::<f64>::build()
            .class_weight(class_weight)
            .shuffle(false)
            .convergence_policy(SKConvergencePolicy::Permissive)
            .build()
            .unwrap()
            .fit(features.view(), labels.view())
            .unwrap()
    };
    let balanced = fit(SKClassWeight::Balanced);
    let explicit = fit(SKClassWeight::Explicit(vec![
        balanced_weight,
        minority_weight,
    ]));
    assert_close(
        explicit.coefficients().as_slice().unwrap(),
        balanced.coefficients().as_slice().unwrap(),
        1e-12,
    );
    let reversed = fit(SKClassWeight::Explicit(vec![
        minority_weight,
        balanced_weight,
    ]));
    assert!(
        reversed
            .coefficients()
            .iter()
            .zip(explicit.coefficients().iter())
            .any(|(left, right)| (left - right).abs() > 1e-9),
        "swapping the class weights must change the fit"
    );
    let plain = fit(SKClassWeight::None);
    let uniform = fit(SKClassWeight::Explicit(vec![1.0, 1.0]));
    assert_close(
        plain.coefficients().as_slice().unwrap(),
        uniform.coefficients().as_slice().unwrap(),
        1e-12,
    );
}

#[test]
fn invalid_explicit_weights_are_rejected() {
    let (features, raw) = imbalanced_binary_design();
    let labels = labels_i64(&raw);
    let negative = SKSGDClassifier::<f64>::build()
        .class_weight(SKClassWeight::Explicit(vec![1.0, -1.0]))
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap_err();
    assert!(
        matches!(negative, SKError::InvalidHyperparameter { name, .. } if name == "class_weight")
    );
    let wrong_length = SKSGDClassifier::<f64>::build()
        .class_weight(SKClassWeight::Explicit(vec![1.0]))
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap_err();
    assert!(matches!(wrong_length, SKError::ShapeMismatch { .. }));
    let all_zero = SKSGDClassifier::<f64>::build()
        .class_weight(SKClassWeight::Explicit(vec![0.0, 0.0]))
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap_err();
    assert!(
        matches!(all_zero, SKError::InvalidHyperparameter { name, .. } if name == "class_weight")
    );
}
