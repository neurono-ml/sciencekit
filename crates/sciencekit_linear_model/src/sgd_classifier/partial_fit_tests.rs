//! Companion tests for incremental learning of the fitted SGD classifier
//! (spec `stochastic-gradient-classifier`).

use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKError, SKSupervisedFit};

use super::SKSGDClassifier;
use crate::sgd::SKConvergencePolicy;
use crate::test_datasets::{labels_i64, separable_binary_design};

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
fn partial_fit_continues_the_schedule_counter() {
    let (features, raw) = separable_binary_design();
    let labels = labels_i64(&raw);
    let mut model = SKSGDClassifier::<f64>::build()
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let before = model.states[0].step;
    let epochs_before = model.epochs_taken();
    model.partial_fit(features.view(), labels.view()).unwrap();
    assert_eq!(model.states[0].step, before + raw.len());
    assert_eq!(model.epochs_taken(), epochs_before + 1);
}

#[test]
fn partial_fit_continues_the_average() {
    let (features, raw) = separable_binary_design();
    let labels = labels_i64(&raw);
    let mut model = SKSGDClassifier::<f64>::build()
        .average(true)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let before = model.states[0].average_count();
    model.partial_fit(features.view(), labels.view()).unwrap();
    assert!(model.states[0].average_count() > before);
    assert!(model.coefficients().iter().all(|value| value.is_finite()));
}

#[test]
fn partial_fit_does_not_increase_the_objective() {
    let (features, raw) = separable_binary_design();
    let labels = labels_i64(&raw);
    let mut model = SKSGDClassifier::<f64>::build()
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let before = mean_hinge(&model, features.view(), &labels.view());
    model.partial_fit(features.view(), labels.view()).unwrap();
    let after = mean_hinge(&model, features.view(), &labels.view());
    assert!(
        after <= before + 1e-9,
        "the partial fit raised the objective from {before} to {after}"
    );
}

#[test]
fn an_unseen_class_is_rejected() {
    let (features, raw) = separable_binary_design();
    let labels = labels_i64(&raw);
    let mut model = SKSGDClassifier::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let unseen = ndarray::Array1::from_vec(vec![0i64, 1, 2, 0, 1, 2]);
    let error = model
        .partial_fit(features.view(), unseen.view())
        .unwrap_err();
    assert!(matches!(error, SKError::InvalidHyperparameter { name, .. } if name == "targets"));
}

#[test]
fn a_column_mismatch_is_rejected() {
    let (features, raw) = separable_binary_design();
    let labels = labels_i64(&raw);
    let mut model = SKSGDClassifier::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let wider = ndarray::Array2::<f64>::zeros((6, 3));
    let error = model.partial_fit(wider.view(), labels.view()).unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}
