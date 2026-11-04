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

/// A binary design whose validation prefix carries the opposite label at the
/// same feature values, so early stopping is guaranteed to trigger.
fn anti_correlated_design() -> (ndarray::Array2<f64>, ndarray::Array1<i64>) {
    let values = [
        0.5, 0.75, 1.0, 1.25, 1.5, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0,
        3.25, 3.5, 3.75, 4.0,
    ];
    let features = ndarray::Array2::from_shape_vec((values.len(), 1), values.to_vec()).unwrap();
    let labels = ndarray::Array1::from_iter(
        values
            .iter()
            .enumerate()
            .map(|(row, _)| if row < 5 { 0 } else { 1 }),
    );
    (features, labels)
}

#[test]
fn partial_fit_after_early_stopping_continues_from_the_best_iterate() {
    use crate::sgd::SKSGDLearningRate;

    let (features, labels) = anti_correlated_design();
    // The validation prefix is the slice early stopping keeps the best iterate
    // on; continuing on it exposes whether the state starts from the reported
    // (best) iterate or from the discarded last one.
    let increment = features.slice(ndarray::s![0..5, ..]).to_owned();
    let increment_labels = labels.slice(ndarray::s![0..5]).to_owned();
    let mut model = SKSGDClassifier::<f64>::build()
        .learning_rate(SKSGDLearningRate::Constant)
        .initial_learning_rate(1e-4)
        .alpha(0.0)
        .shuffle(false)
        .early_stopping(true)
        .validation_fraction(0.25)
        .number_of_epochs_without_improvement(1)
        .maximum_number_of_iterations(10)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    assert!(model.converged());
    let before = mean_hinge(&model, increment.view(), &increment_labels.view());
    model
        .partial_fit(increment.view(), increment_labels.view())
        .unwrap();
    let after = mean_hinge(&model, increment.view(), &increment_labels.view());
    assert!(
        after <= before + 1e-12,
        "the continuation regressed from {before} to {after}"
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
