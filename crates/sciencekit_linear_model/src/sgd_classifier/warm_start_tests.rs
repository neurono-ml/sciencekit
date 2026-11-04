//! Companion tests for the explicit warm start of the SGD classifier (spec
//! `stochastic-gradient-classifier`).

use ndarray::{Array1, Array2};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKError, SKSupervisedFit, sk_canonicalize_labels};

use super::{SKSGDClassifier, SKSGDClassifierCoefficients};
use crate::sgd::{SKConvergencePolicy, SKSGDLearningRate};
use crate::test_datasets::{
    assert_close, labels_i64, non_separable_binary_design, separable_binary_design,
};

fn mean_hinge(
    model: &super::SKSGDClassifierModel<f64>,
    features: ndarray::ArrayView2<'_, f64>,
    labels: &ndarray::ArrayView1<'_, i64>,
) -> f64 {
    let decisions = model.decision_function(features).unwrap();
    let total: f64 = labels
        .iter()
        .enumerate()
        .map(|(row, &label)| {
            let sign = if label == 1 { 1.0 } else { -1.0 };
            (1.0 - sign * decisions[[row, 0]]).max(0.0)
        })
        .sum();
    total / labels.len() as f64
}

#[test]
fn a_seeded_refit_starts_from_the_source() {
    let (features, raw) = separable_binary_design();
    let labels = labels_i64(&raw);
    let source = SKSGDClassifier::<f64>::build()
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let refit = SKSGDClassifier::<f64>::build()
        .warm_start_from_model(&source)
        .learning_rate(SKSGDLearningRate::Constant)
        .initial_learning_rate(1e-6)
        .alpha(0.0)
        .maximum_number_of_iterations(1)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    assert_close(
        refit.coefficients().as_slice().unwrap(),
        source.coefficients().as_slice().unwrap(),
        1e-3,
    );
}

#[test]
fn a_seeded_refit_never_regresses_within_the_same_budget() {
    let (features, raw) = non_separable_binary_design();
    let labels = labels_i64(&raw);
    let source = SKSGDClassifier::<f64>::build()
        .learning_rate(SKSGDLearningRate::Constant)
        .initial_learning_rate(0.05)
        .maximum_number_of_iterations(200)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let cold = SKSGDClassifier::<f64>::build()
        .learning_rate(SKSGDLearningRate::Constant)
        .initial_learning_rate(1e-3)
        .maximum_number_of_iterations(1)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let warm = SKSGDClassifier::<f64>::build()
        .learning_rate(SKSGDLearningRate::Constant)
        .initial_learning_rate(1e-3)
        .warm_start_from_model(&source)
        .maximum_number_of_iterations(1)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let cold_loss = mean_hinge(&cold, features.view(), &labels.view());
    let warm_loss = mean_hinge(&warm, features.view(), &labels.view());
    assert!(
        warm_loss <= cold_loss + 1e-9,
        "the warm refit regressed from {cold_loss} to {warm_loss}"
    );
}

#[test]
fn a_warm_start_without_intercept_starts_from_zero() {
    let (features, raw) = separable_binary_design();
    let labels = labels_i64(&raw);
    let fitted = SKSGDClassifier::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let source = SKSGDClassifierCoefficients::<f64>::new(
        fitted.classes().clone(),
        Array2::zeros((1, 2)).view(),
        Array1::from_vec(vec![7.0]).view(),
    )
    .unwrap();
    let model = SKSGDClassifier::<f64>::build()
        .include_intercept(false)
        .warm_start_from(source)
        .learning_rate(SKSGDLearningRate::Constant)
        .alpha(0.0)
        .shuffle(false)
        .maximum_number_of_iterations(3)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    assert!(model.intercepts().iter().all(|&value| value == 0.0));
    let decisions = model.decision_function(features.view()).unwrap();
    let expected = features.dot(&model.coefficients().row(0));
    assert_close(
        decisions.column(0).to_slice().unwrap(),
        expected.as_slice().unwrap(),
        1e-12,
    );
}

#[test]
fn a_misaligned_feature_count_is_rejected() {
    let (features, raw) = separable_binary_design();
    let labels = labels_i64(&raw);
    let fitted = SKSGDClassifier::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let source = SKSGDClassifierCoefficients::<f64>::new(
        fitted.classes().clone(),
        Array2::zeros((1, 3)).view(),
        Array1::zeros(1).view(),
    )
    .unwrap();
    let error = SKSGDClassifier::<f64>::build()
        .warm_start_from(source)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

#[test]
fn a_misaligned_class_table_is_rejected() {
    let (features, raw) = separable_binary_design();
    let labels = labels_i64(&raw);
    let (_, other_classes) = sk_canonicalize_labels(&["a", "b", "c"]);
    let source = SKSGDClassifierCoefficients::<f64>::new(
        other_classes,
        Array2::zeros((3, 2)).view(),
        Array1::zeros(3).view(),
    )
    .unwrap();
    let error = SKSGDClassifier::<f64>::build()
        .warm_start_from(source)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap_err();
    assert!(
        matches!(error, SKError::InvalidHyperparameter { name, .. } if name == "warm_start_from")
    );
}

#[test]
fn a_wrong_head_count_is_rejected_at_construction() {
    let (_, classes) = sk_canonicalize_labels(&["a", "b", "c"]);
    let error = SKSGDClassifierCoefficients::<f64>::new(
        classes,
        Array2::zeros((1, 2)).view(),
        Array1::zeros(1).view(),
    )
    .unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}
