//! Companion tests for the one-vs-rest multi-class fit (spec
//! `stochastic-gradient-classifier`).

use ndarray::Array2;
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKClassifierPredictor, SKSupervisedFit};

use super::SKSGDClassifier;
use crate::sgd::SKConvergencePolicy;
use crate::sgd::SKSGDClassifierLoss;
use crate::test_datasets::{labels_i64, three_class_design};

#[test]
fn three_classes_produce_three_heads() {
    let (features, raw) = three_class_design();
    let labels = labels_i64(&raw);
    let model = SKSGDClassifier::<f64>::build()
        .shuffle(false)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    assert_eq!(model.number_of_heads(), 3);
    assert_eq!(model.coefficients().nrows(), 3);
    assert_eq!(model.number_of_classes(), 3);
    let predicted = model.predict_labels(features.view()).unwrap();
    assert_eq!(predicted.to_vec(), labels.to_vec());
    let decisions = model.decision_function(features.view()).unwrap();
    assert_eq!(decisions.dim(), (6, 3));
}

#[test]
fn labels_are_the_arg_max_of_the_decisions() {
    let (features, raw) = three_class_design();
    let labels = labels_i64(&raw);
    let model = SKSGDClassifier::<f64>::build()
        .shuffle(false)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let decisions = model.decision_function(features.view()).unwrap();
    let predicted = model.predict_labels(features.view()).unwrap();
    for row in 0..decisions.nrows() {
        let mut best = 0;
        for head in 1..decisions.ncols() {
            if decisions[[row, head]] > decisions[[row, best]] {
                best = head;
            }
        }
        assert_eq!(predicted[row], best as i64);
    }
}

#[test]
fn log_loss_probabilities_sum_to_one() {
    let (features, raw) = three_class_design();
    let labels = labels_i64(&raw);
    let model = SKSGDClassifier::<f64>::build()
        .loss(SKSGDClassifierLoss::LogLoss)
        .convergence_policy(SKConvergencePolicy::Permissive)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let probabilities = model.predict_probabilities(features.view()).unwrap();
    assert_eq!(probabilities.dim(), (6, 3));
    for row in probabilities.rows() {
        let total: f64 = row.sum();
        assert!((total - 1.0).abs() < 1e-9);
        assert!(row.iter().all(|value| (0.0..=1.0).contains(value)));
    }
}

#[test]
fn a_two_class_design_keeps_one_head() {
    let (features, raw) = three_class_design();
    let features = features.slice(ndarray::s![0..4, ..]).to_owned();
    let labels = labels_i64(&raw[0..4]);
    let model = SKSGDClassifier::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    assert_eq!(model.number_of_heads(), 1);
    assert_eq!(model.number_of_classes(), 2);
}

#[test]
fn ties_resolve_to_the_lowest_canonical_index() {
    let features = Array2::<f64>::zeros((6, 1));
    let labels = ndarray::Array1::from_vec(vec![0i64, 0, 1, 1, 2, 2]);
    let model = SKSGDClassifier::<f64>::build()
        .include_intercept(false)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let predicted = model.predict_labels(features.view()).unwrap();
    assert!(predicted.iter().all(|&label| label == 0));
}
