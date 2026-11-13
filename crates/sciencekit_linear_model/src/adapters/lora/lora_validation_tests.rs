//! Companion tests for adapter validation and concurrency (task 3.3).

use std::sync::Arc;
use std::thread;

use ndarray::{Array1, Array2, array};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{SKError, SKRegressorPredictor, SKSupervisedFit};
use sprs::TriMat;

use crate::adapters::lora::{
    SKLoraAdapterBuilder, SKLoraAdapterRegistry, SKLoraAttachedModel, SKLoraBaseModel,
    SKLoraTargetModule,
};
use crate::linear_regression::core_implementation::SKLinearRegression;
use crate::logistic_regression::SKLogisticRegression;

fn linear_dataset() -> (Array2<f64>, Array1<f64>) {
    let features = array![
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [2.0, -1.0],
        [-1.0, 2.0]
    ];
    let targets = features.dot(&array![2.0, -3.0]) + 1.0;
    (features, targets)
}

fn linear_base() -> SKLoraBaseModel<f64> {
    let (features, targets) = linear_dataset();
    let model = SKLinearRegression::build()
        .build()
        .unwrap()
        .fit(features.view(), targets.view())
        .unwrap();
    SKLoraBaseModel::linear_regression(model)
}

fn fitted_regression_adapter(identifier: &str) -> SKLoraAttachedModel<f64> {
    let features = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    let targets = features.dot(&array![2.7, -2.6]) + 1.0;
    let mut attached = SKLoraAdapterBuilder::<f64>::new()
        .identifier(identifier)
        .rank(2)
        .build()
        .unwrap()
        .attach(linear_base())
        .unwrap();
    attached.fit(features.view(), targets.view()).unwrap();
    attached
}

#[test]
fn duplicate_adapter_identifiers_are_rejected() {
    let mut registry = SKLoraAdapterRegistry::<f64>::new();
    registry
        .insert(fitted_regression_adapter("shared"))
        .expect("the first identifier is free");
    let error = registry
        .insert(fitted_regression_adapter("shared"))
        .expect_err("the second insert reuses an identifier");

    match error {
        SKError::InvalidHyperparameter { name, .. } => assert_eq!(name, "identifier"),
        other => panic!("expected InvalidHyperparameter, found {other:?}"),
    }
    assert_eq!(registry.number_of_adapters(), 1);
}

#[test]
fn the_registry_addresses_adapters_by_identifier() {
    let mut registry = SKLoraAdapterRegistry::<f64>::new();
    registry
        .insert(fitted_regression_adapter("billing"))
        .unwrap();
    registry
        .insert(fitted_regression_adapter("support"))
        .unwrap();

    assert_eq!(
        registry.identifiers().collect::<Vec<_>>(),
        vec!["billing", "support"]
    );
    assert!(registry.get("billing").is_some());
    assert!(registry.get("missing").is_none());
    assert!(!registry.is_empty());
}

#[test]
fn a_zero_rank_adapter_is_rejected_before_any_base_is_touched() {
    let error = SKLoraAdapterBuilder::<f64>::new()
        .rank(0)
        .build()
        .expect_err("rank zero is not a low-rank factorization");
    assert!(matches!(error, SKError::InvalidHyperparameter { .. }));
}

#[test]
fn a_tree_base_family_is_rejected_with_a_typed_error() {
    let error = SKLoraAdapterBuilder::<f64>::new()
        .build()
        .unwrap()
        .attach(SKLoraBaseModel::unsupported_family("random_forest"))
        .expect_err("forests have no coefficient block");
    assert!(matches!(error, SKError::UnsupportedOperation { .. }));
}

#[test]
fn the_wrong_target_module_is_rejected_with_a_typed_error() {
    let error = SKLoraAdapterBuilder::<f64>::new()
        .target(SKLoraTargetModule::LogisticCoefficients)
        .build()
        .unwrap()
        .attach(linear_base())
        .expect_err("the base is a regressor");
    assert!(matches!(error, SKError::UnsupportedOperation { .. }));
}

#[test]
fn non_finite_features_are_rejected_at_fit() {
    let mut features = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
    features[[1, 0]] = f64::INFINITY;
    let targets = array![1.0, 2.0, 3.0, 4.0];
    let mut attached = SKLoraAdapterBuilder::<f64>::new()
        .rank(2)
        .build()
        .unwrap()
        .attach(linear_base())
        .unwrap();
    let error = attached
        .fit(features.view(), targets.view())
        .expect_err("infinite features cannot train an adapter");
    assert!(matches!(error, SKError::NonFiniteValue { .. }));
}

#[test]
fn non_finite_sparse_features_are_rejected_at_fit() {
    let mut triplets = TriMat::new((3, 2));
    triplets.add_triplet(0, 0, 1.0);
    triplets.add_triplet(1, 1, f64::NAN);
    let sparse = triplets.to_csr();
    let targets = array![1.0, 2.0, 3.0];
    let mut attached = SKLoraAdapterBuilder::<f64>::new()
        .rank(2)
        .build()
        .unwrap()
        .attach(linear_base())
        .unwrap();
    let error = attached
        .fit(sparse.view(), targets.view())
        .expect_err("a NaN stored in a sparse input cannot train an adapter");
    assert!(matches!(error, SKError::NonFiniteValue { .. }));
}

#[test]
fn an_empty_design_is_rejected_at_fit() {
    let mut attached = SKLoraAdapterBuilder::<f64>::new()
        .rank(1)
        .build()
        .unwrap()
        .attach(linear_base())
        .unwrap();
    let empty_features = Array2::<f64>::zeros((0, 2));
    let empty_targets = Array1::<f64>::zeros(0);
    let error = attached
        .fit(empty_features.view(), empty_targets.view())
        .expect_err("a fit needs at least one row");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

#[test]
fn classification_rejects_a_feature_count_that_disagrees_with_the_base() {
    let features = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
    let labels = array![0_i64, 1, 1];
    let model = SKLogisticRegression::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let attached = SKLoraAdapterBuilder::<f64>::new()
        .rank(1)
        .target(SKLoraTargetModule::LogisticCoefficients)
        .build()
        .unwrap()
        .attach(SKLoraBaseModel::logistic_regression(model))
        .unwrap();
    let error = attached
        .decision_function(array![[0.0, 0.0, 1.0]].view())
        .expect_err("three features cannot score a two-feature base");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

#[test]
fn concurrent_prediction_matches_the_sequential_result() {
    let attached = Arc::new(fitted_regression_adapter("concurrent"));
    let features = Arc::new(array![[0.5, -0.5], [1.5, 1.0], [-1.0, 0.25]]);
    let expected = attached.predict(features.view()).unwrap();

    let handles = (0..8)
        .map(|_| {
            let attached = Arc::clone(&attached);
            let features = Arc::clone(&features);
            thread::spawn(move || attached.predict(features.view()).unwrap())
        })
        .collect::<Vec<_>>();
    for handle in handles {
        let prediction = handle.join().expect("the prediction thread must not panic");
        assert_eq!(prediction, expected);
    }
}

#[test]
fn an_adapter_is_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SKLoraAttachedModel<f64>>();
    assert_send_sync::<SKLoraAdapterRegistry<f64>>();
}
