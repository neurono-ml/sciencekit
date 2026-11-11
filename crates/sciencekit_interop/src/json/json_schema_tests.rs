//! Companion tests for the JSON debug schema (spec `json-debug-format`).

use sciencekit_common::{
    SKError, SKModelMetadata, SKModelState, SKParameterValue, SKTensor, SKTensorDataType,
};

use super::schema::encode_model_state_to_json_string;
use super::{SK_JSON_DEBUG_FORMAT_VERSION, SKJsonFormatMode};

/// Hand-written fixture matching the documented `SKStandardScaler` schema.
const HAND_WRITTEN_STANDARD_SCALER_FIXTURE: &str = r#"{
  "format_version": 1,
  "estimator": "SKStandardScaler",
  "dtype": "f64",
  "feature_count": 4,
  "class_labels": null,
  "hyperparameters": {
    "with_mean": true,
    "with_std": true
  },
  "tensors": {
    "standard_scaler.mean": {
      "dtype": "F64",
      "shape": [
        4
      ],
      "values": [
        1.0,
        2.0,
        3.0,
        4.0
      ]
    }
  }
}"#;

/// Serializing the `SKStandardScaler` state produces the exact documented fixture.
#[test]
fn json_schema_matches_hand_written_standard_scaler_fixture() {
    assert_eq!(SK_JSON_DEBUG_FORMAT_VERSION, 1);

    let metadata = SKModelMetadata::new("SKStandardScaler", 1, SKTensorDataType::F64, 4)
        .unwrap()
        .with_parameter("with_mean", SKParameterValue::from(true))
        .unwrap()
        .with_parameter("with_std", SKParameterValue::from(true))
        .unwrap();
    let mean_tensor = SKTensor::floating(
        "standard_scaler.mean",
        vec![4],
        vec![1.0_f64, 2.0, 3.0, 4.0],
    )
    .unwrap();
    let state = SKModelState::new(metadata, vec![mean_tensor]).unwrap();

    let rendered = encode_model_state_to_json_string(&state, SKJsonFormatMode::Pretty).unwrap();

    assert_eq!(rendered, HAND_WRITTEN_STANDARD_SCALER_FIXTURE);
}

/// Estimator metadata, class labels and every hyperparameter type serialize as plain JSON fields.
#[test]
fn json_schema_renders_readable_metadata_and_typed_hyperparameters() {
    let nested_list = SKParameterValue::List(vec![
        SKParameterValue::from(2_i64),
        SKParameterValue::from("auto"),
    ]);
    let metadata = SKModelMetadata::new("SKLogisticRegression", 1, SKTensorDataType::F32, 3)
        .unwrap()
        .with_class_labels(vec!["setosa".to_string(), "versicolor".to_string()])
        .unwrap()
        .with_parameter("fit_intercept", SKParameterValue::from(true))
        .unwrap()
        .with_parameter(
            "maximum_number_of_iterations",
            SKParameterValue::from(200_i64),
        )
        .unwrap()
        .with_parameter("tolerance", SKParameterValue::from(1.0e-4_f64))
        .unwrap()
        .with_parameter("solver", SKParameterValue::from("lbfgs"))
        .unwrap()
        .with_parameter("schedule", nested_list)
        .unwrap();
    let weights =
        SKTensor::floating("logistic.weights", vec![1, 3], vec![0.1_f32, -0.25, 0.5]).unwrap();
    let state = SKModelState::new(metadata, vec![weights]).unwrap();

    let rendered = encode_model_state_to_json_string(&state, SKJsonFormatMode::Pretty).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&rendered).unwrap();

    assert_eq!(parsed["format_version"], 1);
    assert_eq!(parsed["estimator"], "SKLogisticRegression");
    assert_eq!(parsed["dtype"], "f32");
    assert_eq!(parsed["feature_count"], 3);
    assert_eq!(
        parsed["class_labels"],
        serde_json::json!(["setosa", "versicolor"])
    );
    assert_eq!(parsed["hyperparameters"]["fit_intercept"], true);
    assert_eq!(
        parsed["hyperparameters"]["maximum_number_of_iterations"],
        200
    );
    assert_eq!(parsed["hyperparameters"]["tolerance"], 1.0e-4);
    assert_eq!(parsed["hyperparameters"]["solver"], "lbfgs");
    assert_eq!(
        parsed["hyperparameters"]["schedule"],
        serde_json::json!([2, "auto"])
    );
    // Single-precision 0.1_f32 formats cleanly as 0.1, not 0.10000000149011612.
    assert!(rendered.contains("0.1,"));
}

/// Floating, integer and boolean tensors expose their shape and flat values directly.
#[test]
fn json_schema_renders_all_tensor_data_types_and_shapes_as_flat_arrays() {
    let metadata =
        SKModelMetadata::new("SKDecisionTreeClassifier", 1, SKTensorDataType::F64, 2).unwrap();
    let thresholds =
        SKTensor::floating("tree.thresholds", vec![2, 2], vec![0.5_f64, 1.5, 2.5, 3.5]).unwrap();
    let scalar_bias = SKTensor::floating("tree.root_impurity", vec![], vec![0.42_f64]).unwrap();
    let feature_indices =
        SKTensor::<f64>::integer("tree.feature_indices", vec![3], vec![0_i64, 1, -1]).unwrap();
    let leaf_flags =
        SKTensor::<f64>::boolean("tree.is_leaf", vec![3], vec![false, false, true]).unwrap();
    let state = SKModelState::new(
        metadata,
        vec![thresholds, scalar_bias, feature_indices, leaf_flags],
    )
    .unwrap();

    let rendered = encode_model_state_to_json_string(&state, SKJsonFormatMode::Compact).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&rendered).unwrap();

    assert_eq!(
        parsed["tensors"]["tree.thresholds"],
        serde_json::json!({ "dtype": "F64", "shape": [2, 2], "values": [0.5, 1.5, 2.5, 3.5] })
    );
    assert_eq!(
        parsed["tensors"]["tree.root_impurity"],
        serde_json::json!({ "dtype": "F64", "shape": [], "values": [0.42] })
    );
    assert_eq!(
        parsed["tensors"]["tree.feature_indices"],
        serde_json::json!({ "dtype": "I64", "shape": [3], "values": [0, 1, -1] })
    );
    assert_eq!(
        parsed["tensors"]["tree.is_leaf"],
        serde_json::json!({ "dtype": "Bool", "shape": [3], "values": [false, false, true] })
    );
}

/// Non-finite floating values cannot be represented in RFC 8259 JSON and fail up front.
#[test]
fn json_schema_rejects_non_finite_values_before_serialization() {
    let metadata = SKModelMetadata::new("SKStandardScaler", 1, SKTensorDataType::F64, 3).unwrap();
    let bad_tensor = SKTensor::floating(
        "standard_scaler.mean",
        vec![3],
        vec![1.0_f64, f64::NAN, 3.0],
    )
    .unwrap();
    let state = SKModelState::new(metadata, vec![bad_tensor]).unwrap();

    let error = encode_model_state_to_json_string(&state, SKJsonFormatMode::Pretty).unwrap_err();
    assert!(matches!(error, SKError::NonFiniteValue { position: 1 }));
}
