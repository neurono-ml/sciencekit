//! Companion tests for the shared class-weighting policy (spec
//! `stochastic-gradient-solver`, decision 11).

use super::policy::SKClassWeight;
use super::sk_expand_class_weights;
use sciencekit_common::SKError;

#[test]
fn none_weighting_weighs_every_sample_one() {
    let weights = sk_expand_class_weights(&SKClassWeight::<f64>::None, &[0, 1, 1], 2).unwrap();
    assert_eq!(weights, vec![1.0, 1.0, 1.0]);
}

#[test]
fn balanced_weighting_equalizes_the_class_totals() {
    // Class 0 has one sample, class 1 has three: the balanced weights must make
    // each class's total weight equal.
    let weights =
        sk_expand_class_weights(&SKClassWeight::<f64>::Balanced, &[0, 1, 1, 1], 2).unwrap();
    let class_zero: f64 = weights[0];
    let class_one: f64 = weights[1..].iter().sum();
    assert!(
        (class_zero - class_one).abs() < 1e-12,
        "balanced weights must equalize the class totals, got {class_zero} and {class_one}"
    );
}

#[test]
fn explicit_weighting_is_applied_in_class_order() {
    let weights =
        sk_expand_class_weights(&SKClassWeight::Explicit(vec![2.0, 5.0]), &[1, 0, 1], 2).unwrap();
    assert_eq!(weights, vec![5.0, 2.0, 5.0]);
}

#[test]
fn negative_and_all_zero_explicit_weights_are_rejected() {
    let negative = sk_expand_class_weights(&SKClassWeight::Explicit(vec![1.0, -1.0]), &[0], 2);
    assert!(matches!(
        negative.unwrap_err(),
        SKError::InvalidHyperparameter { name, .. } if name == "class_weight"
    ));
    let all_zero = sk_expand_class_weights(&SKClassWeight::Explicit(vec![0.0, 0.0]), &[0], 2);
    assert!(all_zero.is_err());
}

#[test]
fn explicit_weight_length_must_match_the_class_count() {
    let error = sk_expand_class_weights(&SKClassWeight::Explicit(vec![1.0]), &[0], 3).unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}
