//! Companion tests for the out-of-core logistic fit (spec
//! `logistic-regression`).

use ndarray::{Array1, Array2};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKClassifierPredictor, SKSupervisedFit};

use super::SKLogisticRegression;
use crate::logistic::test_datasets::{SupervisedFloatSource, labels_i64, separable_binary_design};

/// The shared binary design split into two float-coded batches.
fn batches() -> Vec<(Array2<f64>, Array1<f64>)> {
    let (features, labels) = separable_binary_design();
    let split = 3;
    let first_features = features.slice(ndarray::s![0..split, ..]).to_owned();
    let first_labels = Array1::from_vec(labels[0..split].iter().map(|&l| l as f64).collect());
    let second_features = features.slice(ndarray::s![split.., ..]).to_owned();
    let second_labels = Array1::from_vec(labels[split..].iter().map(|&l| l as f64).collect());
    vec![
        (first_features, first_labels),
        (second_features, second_labels),
    ]
}

#[test]
fn streaming_equals_the_dense_fit() {
    let (all_features, raw) = separable_binary_design();
    let all_labels = labels_i64(&raw);
    let dense = SKLogisticRegression::<f64>::build()
        .build()
        .unwrap()
        .fit(all_features.view(), all_labels.view())
        .unwrap();
    let mut source = SupervisedFloatSource::new(batches());
    let streamed = SKLogisticRegression::<f64>::build()
        .build()
        .unwrap()
        .fit_streaming(&mut source)
        .unwrap();
    for (dense_value, streamed_value) in dense.coefficients().iter().zip(streamed.coefficients()) {
        assert!((dense_value - streamed_value).abs() < 1e-6);
    }
    let predicted = streamed.predict_labels(all_features.view()).unwrap();
    assert_eq!(predicted.to_vec(), all_labels.to_vec());
}

#[test]
fn an_empty_source_is_rejected() {
    let mut source = SupervisedFloatSource::<f64>::new(Vec::new());
    let result = SKLogisticRegression::<f64>::build()
        .build()
        .unwrap()
        .fit_streaming(&mut source);
    assert!(result.is_err());
}
