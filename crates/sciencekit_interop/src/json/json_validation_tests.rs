//! Companion validation tests for the JSON debug reader (spec `json-debug-format`).

use sciencekit_common::SKError;

use super::sk_read_model_state_json;

fn document_with_tensors(scalar_data_type: &str, tensors_body: &str) -> String {
    format!(
        r#"{{
            "format_version": 1,
            "estimator": "SKMock",
            "dtype": "{scalar_data_type}",
            "feature_count": 1,
            "class_labels": null,
            "hyperparameters": {{}},
            "tensors": {{ {tensors_body} }}
        }}"#
    )
}

/// Malformed JSON syntax, wrong root types and missing required v1 fields return `Conversion`.
#[test]
fn json_validation_rejects_malformed_json_with_conversion_error() {
    let missing_tensors = r#"{
        "format_version": 1,
        "estimator": "SKStandardScaler",
        "dtype": "f64",
        "feature_count": 1,
        "class_labels": null,
        "hyperparameters": {}
    }"#;
    let trailing = format!("{} trailing", document_with_tensors("f64", ""));
    let cases = [
        "",
        "not json at all",
        "{ \"format_version\": 1, ",
        "[1, 2, 3]",
        r#"{"format_version": "one"}"#,
        missing_tensors,
        trailing.as_str(),
    ];

    for malformed in cases {
        let error = sk_read_model_state_json::<f64>(malformed).unwrap_err();
        assert!(
            matches!(error, SKError::Conversion(_)),
            "expected Conversion for `{malformed}`, got {error:?}"
        );
    }
}

/// Any major version other than 1 is rejected as `UnsupportedRepresentation` before v1 checks.
#[test]
fn json_validation_rejects_unknown_major_versions() {
    let full_version_two = r#"{"format_version": 2, "estimator": "SKMock", "dtype": "f64", "feature_count": 1, "class_labels": null, "hyperparameters": {}, "tensors": {}}"#;
    let full_version_zero = r#"{"format_version": 0, "estimator": "SKMock", "dtype": "f64", "feature_count": 1, "class_labels": null, "hyperparameters": {}, "tensors": {}}"#;
    let evolved_version_two = r#"{"format_version": 2, "manifest": {}}"#;
    let incompatible_tensors_version_two = r#"{"format_version": 2, "tensors": []}"#;

    for unsupported in [
        full_version_two,
        full_version_zero,
        evolved_version_two,
        incompatible_tensors_version_two,
    ] {
        let error = sk_read_model_state_json::<f64>(unsupported).unwrap_err();
        assert!(
            matches!(error, SKError::UnsupportedRepresentation { .. }),
            "expected UnsupportedRepresentation for `{unsupported}`, got {error:?}"
        );
    }
}

/// Duplicate tensor keys in the JSON object are detected instead of silently overwritten.
#[test]
fn json_validation_rejects_duplicate_tensor_and_hyperparameter_names() {
    let duplicate_tensors = document_with_tensors(
        "f64",
        r#"
        "standard_scaler.mean": { "dtype": "F64", "shape": [1], "values": [1.0] },
        "standard_scaler.mean": { "dtype": "F64", "shape": [1], "values": [2.0] }
        "#,
    );
    assert!(matches!(
        sk_read_model_state_json::<f64>(&duplicate_tensors),
        Err(SKError::InvalidHyperparameter {
            name: "tensor_names",
            ..
        })
    ));

    let duplicate_hyperparameters = r#"{
        "format_version": 1,
        "estimator": "SKStandardScaler",
        "dtype": "f64",
        "feature_count": 1,
        "class_labels": null,
        "hyperparameters": { "with_mean": true, "with_mean": false },
        "tensors": {}
    }"#;
    assert!(matches!(
        sk_read_model_state_json::<f64>(duplicate_hyperparameters),
        Err(SKError::InvalidHyperparameter {
            name: "hyperparameters",
            ..
        })
    ));
}

/// Invalid scalar/tensor dtypes and mismatched element types are rejected with structured errors.
#[test]
fn json_validation_rejects_invalid_and_mismatched_dtypes() {
    assert!(matches!(
        sk_read_model_state_json::<f64>(&document_with_tensors("f16", "")),
        Err(SKError::InvalidHyperparameter { name: "dtype", .. })
    ));
    assert!(matches!(
        sk_read_model_state_json::<f64>(&document_with_tensors("f32", "")),
        Err(SKError::InvalidHyperparameter { name: "dtype", .. })
    ));

    let bad_tensor_cases = [
        r#""weight": {"dtype": "U8", "shape": [1], "values": [1]}"#,
        r#""weight": {"dtype": "F32", "shape": [1], "values": [1.0]}"#,
        r#""weight": {"dtype": "I64", "shape": [1], "values": [1.5]}"#,
        r#""weight": {"dtype": "Bool", "shape": [1], "values": [1]}"#,
        r#""weight": {"dtype": "F64", "shape": [1], "values": ["nan"]}"#,
    ];

    for snippet in bad_tensor_cases {
        let document = document_with_tensors("f64", snippet);
        let error = sk_read_model_state_json::<f64>(&document).unwrap_err();
        assert!(matches!(
            error,
            SKError::InvalidHyperparameter {
                name: "tensor_data_type",
                ..
            }
        ));
    }

    let overflow_single_precision = document_with_tensors(
        "f32",
        r#""weight": { "dtype": "F32", "shape": [1], "values": [1.0e300] }"#,
    );
    assert!(matches!(
        sk_read_model_state_json::<f32>(&overflow_single_precision),
        Err(SKError::NonFiniteValue { position: 0 })
    ));
}

/// Any disagreement between a tensor's shape product and flat value length returns `ShapeMismatch`.
#[test]
fn json_validation_rejects_tensor_shape_mismatches() {
    let overflow_shape = format!(
        r#""weight": {{"dtype": "F64", "shape": [{}, 2], "values": [1.0]}}"#,
        usize::MAX
    );
    let shape_cases = [
        r#""weight": {"dtype": "F64", "shape": [4], "values": [1.0, 2.0, 3.0]}"#,
        r#""weight": {"dtype": "F64", "shape": [2, 2], "values": [1.0, 2.0, 3.0, 4.0, 5.0]}"#,
        r#""weight": {"dtype": "F64", "shape": [], "values": []}"#,
        overflow_shape.as_str(),
    ];

    for snippet in shape_cases {
        let document = document_with_tensors("f64", snippet);
        let error = sk_read_model_state_json::<f64>(&document).unwrap_err();
        assert!(matches!(error, SKError::ShapeMismatch { .. }));
    }
}
