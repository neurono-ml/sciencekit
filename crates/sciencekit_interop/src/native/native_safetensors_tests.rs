//! Companion tests: Safetensors and JSON states become native models whose
//! predictions match the originals (spec `native-model-import`, requirement
//! "Safetensors native import").

use ndarray::Array2;
use sciencekit_common::{
    SKClassifierPredictor, SKClustererPredictor, SKExportable, SKFeatureTransformer,
    SKRegressorPredictor,
};

use super::conversion::SKNativeImport;
use super::{sk_import_native_model_from_bytes, test_models};
use crate::json::sk_write_model_state_json_pretty;
use crate::safetensors::sk_export_model_safetensors;

fn assert_linear_round_trip(imported: SKNativeImport<f64>, features: &Array2<f64>) {
    let original = test_models::linear_regression();
    let expected = original.predict(features.view()).unwrap();
    let found = imported.predict_regression(features.view()).unwrap();
    assert_eq!(expected, found);
    assert_eq!(imported.descriptor().feature_count(), 2);
    assert_eq!(imported.descriptor().class_labels(), None::<&[String]>);
}

/// Safetensors round trip preserves linear-regression predictions.
#[test]
fn native_safetensors_round_trips_linear_regression() {
    let model = test_models::linear_regression();
    let bytes = sk_export_model_safetensors(&model).unwrap();
    let imported = sk_import_native_model_from_bytes::<f64>(&bytes).unwrap();
    assert_linear_round_trip(imported, &test_models::features());
}

/// JSON round trip preserves linear-regression predictions.
#[test]
fn native_json_round_trips_linear_regression() {
    let model = test_models::linear_regression();
    let state = sciencekit_common::SKExportable::export_state(&model).unwrap();
    let json = sk_write_model_state_json_pretty(&state).unwrap();
    let imported = sk_import_native_model_from_bytes::<f64>(json.as_bytes()).unwrap();
    assert_linear_round_trip(imported, &test_models::features());
}

/// Safetensors round trip preserves logistic labels and probabilities.
#[test]
fn native_safetensors_round_trips_logistic_regression() {
    let model = test_models::logistic_regression();
    let bytes = sk_export_model_safetensors(&model).unwrap();
    let imported = sk_import_native_model_from_bytes::<f64>(&bytes).unwrap();
    let features = test_models::features();
    assert_eq!(
        model.predict_labels(features.view()).unwrap(),
        imported.predict_labels(features.view()).unwrap()
    );
    assert_eq!(
        model.predict_probabilities(features.view()).unwrap(),
        imported.predict_probabilities(features.view()).unwrap()
    );
    assert_eq!(imported.descriptor().class_labels().unwrap().len(), 2);
    assert!(imported.supports_retraining());
}

/// JSON round trip preserves the scaler's transformation.
#[test]
fn native_json_round_trips_standard_scaler() {
    let model = test_models::standard_scaler();
    let state = sciencekit_common::SKExportable::export_state(&model).unwrap();
    let json = sk_write_model_state_json_pretty(&state).unwrap();
    let imported = sk_import_native_model_from_bytes::<f64>(json.as_bytes()).unwrap();
    let features = test_models::features();
    assert_eq!(
        model.transform(features.view()).unwrap(),
        imported.transform(features.view()).unwrap()
    );
    assert!(!imported.supports_retraining());
}

/// Safetensors round trip preserves a fitted tree and a fitted forest.
#[test]
fn native_safetensors_round_trips_tree_and_forest() {
    let tree = test_models::decision_tree_regressor();
    let bytes = sk_export_model_safetensors(&tree).unwrap();
    let imported = sk_import_native_model_from_bytes::<f64>(&bytes).unwrap();
    let features = test_models::features();
    assert_eq!(
        tree.predict(features.view()).unwrap(),
        imported.predict_regression(features.view()).unwrap()
    );
    assert!(!imported.supports_retraining());

    let forest = test_models::random_forest_classifier();
    let bytes = sk_export_model_safetensors(&forest).unwrap();
    let imported = sk_import_native_model_from_bytes::<f64>(&bytes).unwrap();
    assert_eq!(
        forest.predict_labels(features.view()).unwrap(),
        imported.predict_labels(features.view()).unwrap()
    );
    assert_eq!(
        forest.predict_probabilities(features.view()).unwrap(),
        imported.predict_probabilities(features.view()).unwrap()
    );
}

/// A leaf threshold of zero routes exactly like the NaN the exporter used
/// before the change: traversal only reads a threshold on split nodes.
#[test]
fn native_safetensors_leaf_threshold_zero_is_lossless() {
    use sciencekit_common::{
        SKModelState, SKTensor, sk_read_floating_tensor, sk_read_integer_tensor,
    };

    let tree = test_models::decision_tree_regressor();
    let state = tree.export_state().unwrap();
    let split = sk_read_integer_tensor(&state, "tree.nodes.split_feature").unwrap();
    let (_, thresholds) = sk_read_floating_tensor(&state, "tree.nodes.threshold").unwrap();
    assert!(
        split
            .iter()
            .zip(&thresholds)
            .filter(|(feature, _)| **feature < 0)
            .all(|(_, threshold)| *threshold == 0.0),
        "the exporter must encode every leaf threshold as zero"
    );

    let mut nan_tensors = Vec::new();
    for (name, tensor) in state.tensors() {
        if name == "tree.nodes.threshold" {
            let values: Vec<f64> = split
                .iter()
                .zip(&thresholds)
                .map(
                    |(feature, threshold)| {
                        if *feature < 0 { f64::NAN } else { *threshold }
                    },
                )
                .collect();
            nan_tensors
                .push(SKTensor::floating(name.clone(), tensor.shape().to_vec(), values).unwrap());
        } else {
            nan_tensors.push(tensor.clone());
        }
    }
    let nan_state = SKModelState::new(state.metadata().clone(), nan_tensors).unwrap();
    let nan_imported = super::sk_import_native_state(&nan_state).unwrap();
    let zero_imported = super::sk_import_native_state(&state).unwrap();
    let probe = ndarray::array![
        [0.0_f64, 0.0],
        [1.0, 0.0],
        [2.0, 1.0],
        [3.0, 2.0],
        [4.0, 3.0],
    ];
    let expected = tree.predict(probe.view()).unwrap();
    assert_eq!(
        expected,
        nan_imported.predict_regression(probe.view()).unwrap()
    );
    assert_eq!(
        expected,
        zero_imported.predict_regression(probe.view()).unwrap()
    );
}

/// Safetensors round trip preserves the k-means assignments.
#[test]
fn native_safetensors_round_trips_kmeans() {
    let model = test_models::kmeans();
    let bytes = sk_export_model_safetensors(&model).unwrap();
    let imported = sk_import_native_model_from_bytes::<f64>(&bytes).unwrap();
    let features = test_models::features();
    assert_eq!(
        model.predict_clusters(features.view()).unwrap(),
        imported.predict_clusters(features.view()).unwrap()
    );
}
