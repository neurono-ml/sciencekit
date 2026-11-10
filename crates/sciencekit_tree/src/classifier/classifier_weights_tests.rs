//! Class-weighting tests for the classifier (task 4.4).

use ndarray::array;
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKError, SKSupervisedFit};

use crate::class_weight::SKClassWeight;

use super::SKDecisionTreeClassifier;

/// An imbalanced leaf: three class-0 samples and one class-1 sample.
fn imbalanced() -> (ndarray::Array2<f64>, ndarray::Array1<i64>) {
    (array![[0.0], [1.0], [2.0], [3.0]], array![0_i64, 0, 0, 1])
}

/// The unweighted leaf reports the empirical class frequencies.
#[test]
fn unweighted_leaf_uses_empirical_frequencies() {
    let (features, labels) = imbalanced();
    let model = SKDecisionTreeClassifier::<f64>::build()
        .maximum_depth(0)
        .build()
        .expect("builds")
        .fit(&features, labels.view())
        .expect("fit succeeds");
    let probabilities = model
        .predict_probabilities(features.view())
        .expect("predict succeeds");
    assert!((probabilities[[0, 0]] - 0.75).abs() < 1e-12);
    assert!((probabilities[[0, 1]] - 0.25).abs() < 1e-12);
}

/// Balanced weights give the two classes equal total mass, so the leaf
/// distribution becomes uniform.
#[test]
fn balanced_weights_equalize_the_class_mass() {
    let (features, labels) = imbalanced();
    let model = SKDecisionTreeClassifier::<f64>::build()
        .maximum_depth(0)
        .class_weight(SKClassWeight::Balanced)
        .build()
        .expect("builds")
        .fit(&features, labels.view())
        .expect("fit succeeds");
    let probabilities = model
        .predict_probabilities(features.view())
        .expect("predict succeeds");
    assert!((probabilities[[0, 0]] - 0.5).abs() < 1e-12);
    assert!((probabilities[[0, 1]] - 0.5).abs() < 1e-12);
}

/// Explicit weights can flip the predicted majority of a leaf.
#[test]
fn explicit_weights_can_flip_the_majority_class() {
    let features = array![[0.0], [1.0], [2.0]];
    let labels = array![0_i64, 0, 1];
    let unweighted = SKDecisionTreeClassifier::<f64>::build()
        .maximum_depth(0)
        .build()
        .expect("builds")
        .fit(&features, labels.view())
        .expect("fit succeeds");
    let weighted = SKDecisionTreeClassifier::<f64>::build()
        .maximum_depth(0)
        .class_weight(SKClassWeight::Explicit(vec![1.0, 3.0]))
        .build()
        .expect("builds")
        .fit(&features, labels.view())
        .expect("fit succeeds");
    assert_eq!(
        unweighted.predict_labels(features.view()).expect("predict"),
        array![0_i64, 0, 0]
    );
    assert_eq!(
        weighted.predict_labels(features.view()).expect("predict"),
        array![1_i64, 1, 1]
    );
    let probabilities = weighted
        .predict_probabilities(features.view())
        .expect("predict succeeds");
    assert!((probabilities[[0, 0]] - 0.4).abs() < 1e-12);
    assert!((probabilities[[0, 1]] - 0.6).abs() < 1e-12);
}

/// An explicit weight vector whose length disagrees with the observed class
/// count is rejected at fit.
#[test]
fn explicit_weights_must_match_the_class_count() {
    let (features, labels) = imbalanced();
    let error = SKDecisionTreeClassifier::<f64>::build()
        .class_weight(SKClassWeight::Explicit(vec![1.0, 1.0, 1.0]))
        .build()
        .expect("builds")
        .fit(&features, labels.view())
        .expect_err("mismatched weights must fail");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// Class weighting is generic over both supported floats.
#[test]
fn class_weights_accept_either_float() {
    fn probability<F: sciencekit_common::SKFloat>(weight: F) -> F {
        let features = ndarray::Array2::from_shape_vec(
            (3, 1),
            [0.0, 1.0, 2.0]
                .iter()
                .map(|value| F::from(*value).expect("fits"))
                .collect(),
        )
        .expect("shape");
        let labels = ndarray::Array1::from_vec(vec![0_i64, 0, 1]);
        let model = SKDecisionTreeClassifier::<F>::build()
            .maximum_depth(0)
            .class_weight(SKClassWeight::Explicit(vec![F::one(), weight]))
            .build()
            .expect("builds")
            .fit(features.view(), labels.view())
            .expect("fit succeeds");
        model
            .predict_probabilities(features.view())
            .expect("predict succeeds")[[0, 1]]
    }
    assert!((probability(1.0_f32) - 1.0 / 3.0).abs() < 1e-6);
    assert!((probability(1.0_f64) - 1.0 / 3.0).abs() < 1e-12);
}
