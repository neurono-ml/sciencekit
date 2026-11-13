//! Companion tests: concurrency, class-table validation and the no-partial-model
//! guarantee (spec `native-model-import`, requirement "Validation and
//! concurrency").

use ndarray::array;
use sciencekit_common::{
    SKClassifierPredictor, SKExportable, SKModelMetadata, SKModelState, SKTensorDataType,
};

use super::{sk_import_native_model_from_bytes, test_models};
use crate::safetensors::sk_export_model_safetensors;

#[test]
fn native_validation_concurrent_conversions_are_identical() {
    let model = test_models::logistic_regression();
    let bytes = sk_export_model_safetensors(&model).unwrap();
    let features = test_models::features();
    let expected = model.predict_probabilities(features.view()).unwrap();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let bytes = &bytes;
                let features = features.clone();
                scope.spawn(move || {
                    let imported = sk_import_native_model_from_bytes::<f64>(bytes).unwrap();
                    imported.predict_probabilities(features.view()).unwrap()
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), expected);
        }
    });
}

#[test]
fn native_validation_rejects_a_classifier_without_a_class_table() {
    let model = test_models::logistic_regression();
    let state = model.export_state().unwrap();
    let metadata = SKModelMetadata::new(
        state.metadata().estimator_name(),
        state.metadata().format_version(),
        SKTensorDataType::F64,
        state.metadata().feature_count(),
    )
    .unwrap();
    let tensors = state.tensors().values().cloned().collect();
    let without_labels = SKModelState::<f64>::new(metadata, tensors).unwrap();
    let error = super::sk_import_native_state(&without_labels).unwrap_err();
    assert!(!error.to_string().is_empty());
}

#[test]
fn native_validation_rejects_a_class_table_of_the_wrong_width() {
    let model = test_models::logistic_regression();
    let state = model.export_state().unwrap();
    let metadata = SKModelMetadata::new(
        state.metadata().estimator_name(),
        state.metadata().format_version(),
        SKTensorDataType::F64,
        state.metadata().feature_count(),
    )
    .unwrap()
    .with_class_labels(vec!["a".to_string(), "b".to_string(), "c".to_string()])
    .unwrap();
    let tensors = state.tensors().values().cloned().collect();
    let broken = SKModelState::<f64>::new(metadata, tensors).unwrap();
    assert!(super::sk_import_native_state(&broken).is_err());
}

#[test]
fn native_validation_rejects_a_state_with_the_wrong_scalar() {
    let linear = test_models::linear_regression();
    let bytes = sk_export_model_safetensors(&linear).unwrap();
    let error = sk_import_native_model_from_bytes::<f32>(&bytes).unwrap_err();
    assert!(!error.to_string().is_empty());
}

#[test]
fn native_validation_a_failed_conversion_leaves_no_model() {
    let model = test_models::linear_regression();
    let state = model.export_state().unwrap();
    let metadata = SKModelMetadata::new(
        state.metadata().estimator_name(),
        state.metadata().format_version(),
        SKTensorDataType::F64,
        state.metadata().feature_count(),
    )
    .unwrap();
    let broken = SKModelState::<f64>::new(metadata, Vec::new()).unwrap();
    assert!(super::sk_import_native_state(&broken).is_err());

    // The same producer, intact, still converts: a rejected attempt has no
    // side effect on the importer.
    let imported = super::sk_import_native_state(&state).unwrap();
    let features = array![[0.0_f64, 1.0]];
    assert!(
        imported
            .predict_regression(features.view())
            .unwrap()
            .iter()
            .all(|value| value.is_finite())
    );
}
