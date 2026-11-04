//! Companion tests for the binary one-vs-rest fit (spec
//! `stochastic-gradient-classifier`).

use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKClassifierPredictor, SKError, SKSupervisedFit};

use super::SKSGDClassifier;
use crate::sgd::SKConvergencePolicy;
use crate::test_datasets::{assert_close, labels_i64, separable_binary_design, to_csr, to_f32};

#[test]
fn a_separable_design_is_learned_exactly() {
    let (features, raw) = separable_binary_design();
    let labels = labels_i64(&raw);
    let model = SKSGDClassifier::<f64>::build()
        .shuffle(false)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    assert_eq!(model.number_of_heads(), 1);
    assert_eq!(model.number_of_classes(), 2);
    let predicted = model.predict_labels(features.view()).unwrap();
    assert_eq!(predicted.to_vec(), labels.to_vec());
}

#[test]
fn decisions_order_the_labels() {
    let (features, raw) = separable_binary_design();
    let labels = labels_i64(&raw);
    let model = SKSGDClassifier::<f64>::build()
        .shuffle(false)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let decisions = model.decision_function(features.view()).unwrap();
    assert_eq!(decisions.ncols(), 1);
    let predicted = model.predict_labels(features.view()).unwrap();
    for (row, &label) in predicted.iter().enumerate() {
        let expected = i64::from(decisions[[row, 0]] > 0.0);
        assert_eq!(label, expected);
    }
}

#[test]
fn dense_and_csr_fits_agree() {
    let (features, raw) = separable_binary_design();
    let labels = labels_i64(&raw);
    let sparse = to_csr(&features);
    let dense_model = SKSGDClassifier::<f64>::build()
        .shuffle(false)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let sparse_model = SKSGDClassifier::<f64>::build()
        .shuffle(false)
        .build()
        .unwrap()
        .fit(sparse.view(), labels.view())
        .unwrap();
    assert_close(
        dense_model.coefficients().as_slice().unwrap(),
        sparse_model.coefficients().as_slice().unwrap(),
        1e-12,
    );
    assert_close(
        dense_model.intercepts().as_slice().unwrap(),
        sparse_model.intercepts().as_slice().unwrap(),
        1e-12,
    );
}

#[test]
fn nominal_labels_round_trip() {
    let (features, _) = separable_binary_design();
    let labels = ["no", "no", "no", "yes", "yes", "yes"];
    let model = SKSGDClassifier::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), &labels[..])
        .unwrap();
    assert_eq!(model.classes().label_of(0), Some("no"));
    assert_eq!(model.classes().label_of(1), Some("yes"));
    let predicted = model.predict_labels(features.view()).unwrap();
    let round_tripped: Vec<&str> = predicted
        .iter()
        .map(|&index| {
            model
                .classes()
                .label_of(index as usize)
                .expect("every predicted index is in the class table")
        })
        .collect();
    assert_eq!(round_tripped, labels);
}

#[test]
fn continuous_and_single_class_targets_are_rejected() {
    let (features, raw) = separable_binary_design();
    let continuous = ndarray::Array1::from_vec(vec![0.1, 0.2, 0.3, 0.9, 0.8, 0.7]);
    let error = SKSGDClassifier::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), continuous.view())
        .unwrap_err();
    assert!(matches!(error, SKError::UnsupportedRepresentation { .. }));
    let single = ndarray::Array1::from_elem(raw.len(), 0i64);
    let error = SKSGDClassifier::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), single.view())
        .unwrap_err();
    assert!(matches!(error, SKError::InvalidHyperparameter { name, .. } if name == "targets"));
}

#[test]
fn a_label_count_mismatch_is_rejected() {
    let (features, raw) = separable_binary_design();
    let labels = ndarray::Array1::from_vec(raw[..4].iter().map(|&v| v as i64).collect());
    let error = SKSGDClassifier::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap_err();
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

#[test]
fn the_policy_decides_what_an_exhausted_budget_does() {
    let (features, raw) = separable_binary_design();
    let labels = labels_i64(&raw);
    let strict = SKSGDClassifier::<f64>::build()
        .shuffle(false)
        .maximum_number_of_iterations(2)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap_err();
    assert!(matches!(strict, SKError::NotConverged { iterations } if iterations == 2));

    let permissive = SKSGDClassifier::<f64>::build()
        .shuffle(false)
        .maximum_number_of_iterations(2)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    assert!(!permissive.converged());
    assert_eq!(permissive.epochs_taken(), 2);
}

#[test]
fn an_f32_fit_predicts() {
    let (features, raw) = separable_binary_design();
    let labels = labels_i64(&raw);
    let features = to_f32(&features);
    let model = SKSGDClassifier::<f32>::build()
        .shuffle(false)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let predicted = model.predict_labels(features.view()).unwrap();
    assert_eq!(predicted.to_vec(), labels.to_vec());
    assert!(model.coefficients().iter().all(|value| value.is_finite()));
}

#[test]
fn the_same_seed_reproduces_and_the_model_serves_threads() {
    let (features, raw) = separable_binary_design();
    let labels = labels_i64(&raw);
    let first = SKSGDClassifier::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let second = SKSGDClassifier::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    assert_eq!(first.coefficients(), second.coefficients());
    let expected = first.predict_labels(features.view()).unwrap();
    std::thread::scope(|scope| {
        let model = &first;
        let features = features.view();
        let expected = &expected;
        for _ in 0..4 {
            scope.spawn(move || {
                let predicted = model.predict_labels(features).unwrap();
                assert_eq!(&predicted, expected);
            });
        }
    });
}
