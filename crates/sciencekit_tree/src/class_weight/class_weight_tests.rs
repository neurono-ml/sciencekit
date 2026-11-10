//! Per-sample expansion tests for the class-weighting policy.

use super::{SKClassWeight, sk_expand_class_weights};

/// The default policy weighs every sample one.
#[test]
fn none_weighs_every_sample_one() {
    let weights =
        sk_expand_class_weights(&SKClassWeight::<f64>::None, &[0, 1, 1], 2).expect("expands");
    assert_eq!(weights, vec![1.0, 1.0, 1.0]);
}

/// Balanced weighting follows `n_samples / (n_classes · class_count)`.
#[test]
fn balanced_matches_the_closed_form() {
    // 6 samples, 2 classes, class 0 appears 4 times and class 1 twice.
    let weights = sk_expand_class_weights(&SKClassWeight::<f64>::Balanced, &[0, 0, 0, 0, 1, 1], 2)
        .expect("expands");
    assert!((weights[0] - 6.0 / 8.0).abs() < 1e-12);
    assert!((weights[4] - 6.0 / 4.0).abs() < 1e-12);
}

/// Explicit weights are selected by canonical class index.
#[test]
fn explicit_weights_are_selected_by_class() {
    let weights = sk_expand_class_weights(&SKClassWeight::Explicit(vec![2.0, 5.0]), &[0, 1, 1], 2)
        .expect("expands");
    assert_eq!(weights, vec![2.0, 5.0, 5.0]);
}

/// A wrong-length explicit vector is a shape mismatch.
#[test]
fn explicit_weights_require_the_class_count() {
    let error = sk_expand_class_weights(&SKClassWeight::Explicit(vec![1.0]), &[0, 1], 2)
        .expect_err("wrong length must fail");
    assert!(matches!(
        error,
        sciencekit_common::SKError::ShapeMismatch { .. }
    ));
}
