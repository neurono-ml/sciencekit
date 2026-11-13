//! Companion tests: an ONNX file produced by another runtime imports natively
//! and matches its onnxruntime output (spec `native-model-import`, requirement
//! "ONNX native import").
//!
//! The fixtures under `validation/external/` are regenerated and checked by
//! `validation/compare_native_import.py`: that script runs onnxruntime, this
//! test runs the native importer over the same bytes and the same expected
//! values.

use std::path::PathBuf;

use ndarray::Array2;
use serde_json::Value;

use super::SKOnnxNativeImporter;

fn fixture_directory() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("validation")
        .join("external")
}

fn read_json(path: &std::path::Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).expect("the fixture is readable"))
        .expect("the fixture is valid JSON")
}

fn read_matrix(path: &std::path::Path) -> Array2<f64> {
    let document = read_json(path);
    let rows: Vec<Vec<f64>> = document["input"]
        .as_array()
        .expect("input.json carries a matrix")
        .iter()
        .map(|row| {
            row.as_array()
                .expect("each row is an array")
                .iter()
                .map(|cell| cell.as_f64().expect("cells are numbers"))
                .collect()
        })
        .collect();
    let height = rows.len();
    let width = rows.first().map(Vec::len).unwrap_or(0);
    Array2::from_shape_vec((height, width), rows.concat()).expect("the matrix is rectangular")
}

fn tolerance(expected: &Value) -> f64 {
    expected["tolerance"].as_f64().unwrap_or(1e-6)
}

fn maximum_difference(expected: &[f64], actual: &[f64]) -> f64 {
    expected
        .iter()
        .zip(actual)
        .map(|(expected, actual)| (expected - actual).abs())
        .fold(0.0, f64::max)
}

#[test]
fn native_onnx_matches_onnxruntime_for_an_external_regressor() {
    let directory = fixture_directory();
    let model = std::fs::read(directory.join("external_linear_regression.onnx"))
        .expect("the fixture model exists");
    let imported = SKOnnxNativeImporter::builder()
        .model_bytes(model)
        .build::<f64>()
        .unwrap();
    let expected = read_json(&directory.join("external_linear_regression.expected.json"));
    let reference: Vec<f64> = expected["prediction"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|row| {
            row.as_array()
                .map(|row| {
                    row.iter()
                        .map(|v| v.as_f64().unwrap())
                        .collect::<Vec<f64>>()
                })
                .unwrap_or_else(|| vec![row.as_f64().unwrap()])
        })
        .collect();
    let features = read_matrix(&directory.join("external_linear_regression.input.json"));
    let prediction = imported.predict_regression(features.view()).unwrap();
    let actual: Vec<f64> = prediction.iter().copied().collect();
    assert!(maximum_difference(&reference, &actual) <= tolerance(&expected));
}

#[test]
fn native_onnx_matches_onnxruntime_for_an_external_classifier() {
    let directory = fixture_directory();
    let model = std::fs::read(directory.join("external_logistic_regression.onnx"))
        .expect("the fixture model exists");
    let imported = SKOnnxNativeImporter::builder()
        .model_bytes(model)
        .build::<f64>()
        .unwrap();
    let expected = read_json(&directory.join("external_logistic_regression.expected.json"));
    let features = read_matrix(&directory.join("external_logistic_regression.input.json"));
    let probabilities = imported
        .predict_probabilities(features.view())
        .unwrap()
        .iter()
        .copied()
        .collect::<Vec<f64>>();
    let reference: Vec<f64> = expected["probabilities"]
        .as_array()
        .unwrap()
        .iter()
        .flat_map(|row| row.as_array().unwrap().iter())
        .map(|value| value.as_f64().unwrap())
        .collect();
    assert!(maximum_difference(&reference, &probabilities) <= tolerance(&expected));
    let labels = imported.predict_labels(features.view()).unwrap();
    let expected_labels: Vec<i64> = expected["labels"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_i64().unwrap())
        .collect();
    assert_eq!(
        labels.iter().copied().collect::<Vec<i64>>(),
        expected_labels
    );
}
