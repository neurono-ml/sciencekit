//! Acceptance scenarios for tiny/large states, all dtypes, class tables,
//! observability spans and reconstructed model inference (PRD §8.7).

use ndarray::array;
use sciencekit_common::{
    SKModelMetadata, SKModelState, SKParameterValue, SKRegressorPredictor, SKTensor,
    SKTensorDataType, sk_canonicalize_labels,
};
use tracing_subscriber::layer::SubscriberExt;

use super::fixtures::{AcceptanceLinearRegressor, SpanRecorder};
use crate::json::{
    SKJsonFormatMode, sk_export_model_json, sk_import_model_json, sk_read_model_state_json,
    sk_write_model_state_json_compact, sk_write_model_state_json_pretty,
};

/// Tiny zero-tensor states round-trip cleanly for both `f32` and `f64`.
#[test]
fn json_acceptance_tiny_states_round_trip_for_both_scalars() {
    let single_meta =
        SKModelMetadata::new("SKDummyTransformer", 1, SKTensorDataType::F32, 1).unwrap();
    let single_state = SKModelState::<f32>::new(single_meta, Vec::new()).unwrap();
    let single_json = sk_write_model_state_json_pretty(&single_state).unwrap();
    assert_eq!(
        sk_read_model_state_json::<f32>(&single_json).unwrap(),
        single_state
    );

    let double_meta =
        SKModelMetadata::new("SKDummyTransformer", 1, SKTensorDataType::F64, 0).unwrap();
    let double_state = SKModelState::<f64>::new(double_meta, Vec::new()).unwrap();
    let double_json = sk_write_model_state_json_compact(&double_state).unwrap();
    assert_eq!(
        sk_read_model_state_json::<f64>(&double_json).unwrap(),
        double_state
    );
}

/// Large state with canonical class labels, full metadata and all tensor dtypes round-trips.
#[test]
fn json_acceptance_large_state_preserves_all_dtypes_class_table_and_spans() {
    let raw_labels = ["setosa", "versicolor", "virginica", "setosa"];
    let (_encoded, label_table) = sk_canonicalize_labels(&raw_labels);
    let canonical_labels: Vec<String> = label_table.labels().map(str::to_string).collect();

    let metadata = SKModelMetadata::new("SKExtraTreesClassifier", 1, SKTensorDataType::F64, 16)
        .unwrap()
        .with_class_labels(canonical_labels.clone())
        .unwrap()
        .with_parameter("number_of_estimators", SKParameterValue::from(64_i64))
        .unwrap()
        .with_parameter("bootstrap", SKParameterValue::from(false))
        .unwrap()
        .with_parameter("minimum_weight_fraction", SKParameterValue::from(0.0125))
        .unwrap()
        .with_parameter("criterion", SKParameterValue::from("gini"))
        .unwrap()
        .with_parameter(
            "feature_priorities",
            SKParameterValue::List(vec![
                SKParameterValue::from(1_i64),
                SKParameterValue::from(3_i64),
            ]),
        )
        .unwrap();

    let thresholds_data: Vec<f64> = (0..50_000).map(|index| (index as f64) * 0.03125).collect();
    let indices_data: Vec<i64> = (0..10_000).map(|index| (index % 16) - 1).collect();
    let leaves_data: Vec<bool> = (0..10_000).map(|index| index % 3 == 0).collect();

    let thresholds =
        SKTensor::floating("ensemble.thresholds", vec![50, 1_000], thresholds_data).unwrap();
    let indices =
        SKTensor::<f64>::integer("ensemble.feature_indices", vec![10_000], indices_data).unwrap();
    let leaves = SKTensor::<f64>::boolean("ensemble.is_leaf", vec![10_000], leaves_data).unwrap();
    let state = SKModelState::new(metadata, vec![thresholds, indices, leaves]).unwrap();

    let recorder = SpanRecorder::default();
    let subscriber = tracing_subscriber::registry().with(recorder.clone());
    let restored = tracing::subscriber::with_default(subscriber, || {
        let json = sk_write_model_state_json_compact(&state).unwrap();
        sk_read_model_state_json::<f64>(&json).unwrap()
    });

    assert_eq!(restored, state);
    assert_eq!(
        restored.metadata().class_labels(),
        Some(canonical_labels.as_slice())
    );
    let recorded_spans = recorder.names.lock().unwrap().clone();
    assert!(
        recorded_spans
            .iter()
            .any(|name| name == "sciencekit.interop.json.write")
    );
    assert!(
        recorded_spans
            .iter()
            .any(|name| name == "sciencekit.interop.json.read")
    );
}

/// Exporting and importing a regressor reproduces identical predictions and zero MSE drift.
#[test]
fn json_acceptance_imported_regressor_reproduces_predictions_and_zero_error() {
    let fitted = AcceptanceLinearRegressor {
        coefficients: vec![2.5, -1.25, 0.75],
        intercept: 0.5,
    };
    let features = array![[1.0_f64, 2.0, 4.0], [0.0, -4.0, 2.0], [3.0, 1.0, -2.0]];
    let targets = array![3.5_f64, 7.0, 5.25];

    let json = sk_export_model_json(&fitted, SKJsonFormatMode::Pretty).unwrap();
    let imported: AcceptanceLinearRegressor = sk_import_model_json::<f64, _>(&json).unwrap();

    let expected_predictions = fitted.predict(features.view()).unwrap();
    let imported_predictions = imported.predict(features.view()).unwrap();
    assert_eq!(imported_predictions, expected_predictions);

    let mean_squared_error = imported_predictions
        .iter()
        .zip(targets.iter())
        .map(|(predicted, actual)| (predicted - actual).powi(2))
        .sum::<f64>()
        / (targets.len() as f64);
    assert_eq!(mean_squared_error, 0.0);
}
