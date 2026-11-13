//! Runtime validation inside `cargo test` (spec `onnx-export`, requirement
//! "Validation", scenario "Exported graph loads").
//!
//! The authoritative check is the committed harness
//! `validation/compare_with_onnxruntime.py`: `onnx.checker` validates the
//! protobuf and `onnxruntime` recomputes every prediction. This test runs that
//! harness when the host Python has both packages and skips with an explicit
//! message otherwise, so CI without the runtime stays green without pretending
//! to have validated anything.

use std::process::Command;

use ndarray::Array2;
use sciencekit_common::{SKBuilder, SKFeatureTransformer, SKUnsupervisedFit};
use sciencekit_preprocessing::standard_scaler::core_implementation::SKStandardScaler;

use crate::export::SKToOnnx;

fn python_has_onnxruntime() -> bool {
    Command::new("python3")
        .args(["-c", "import onnx, onnxruntime"])
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

#[test]
fn onnx_runtime_validates_an_exported_scaler() {
    if !python_has_onnxruntime() {
        eprintln!(
            "skipped: python3 lacks onnx/onnxruntime; run validation/compare_with_onnxruntime.py in the devcontainer"
        );
        return;
    }
    let features: Array2<f64> = ndarray::array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0], [4.0, 40.0]];
    let model = SKStandardScaler::build()
        .build()
        .expect("the scaler configuration is valid")
        .fit(&features)
        .expect("the scaler fits");
    let transformed = model.transform(&features).expect("the scaler transforms");
    let onnx = model.to_onnx().expect("the scaler exports");
    let directory = tempfile::tempdir().expect("a temporary directory");
    let fixture = directory.path().join("standard_scaler");
    std::fs::create_dir_all(&fixture).expect("the fixture directory");
    write_fixture_at(&fixture, &onnx, &features, &transformed);
    // The harness compares three predictions: Rust, the imported model and
    // onnxruntime. Record the imported one so the harness has all three.
    write_imported_at(&fixture, &onnx, &features);
    let harness = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("validation/compare_with_onnxruntime.py");
    let output = Command::new("python3")
        .arg(harness)
        .arg(&fixture)
        .output()
        .expect("the harness runs");
    assert!(
        output.status.success(),
        "the reference runtime rejected the graph: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

fn write_imported_at(
    directory: &std::path::Path,
    model: &crate::graph::SKOnnxModel,
    features: &Array2<f64>,
) {
    use crate::SKOnnxPredictor;
    let predictor = SKOnnxPredictor::<f64>::builder()
        .model_bytes(model.bytes().to_vec())
        .build()
        .expect("the exported model imports");
    let outputs = predictor
        .predict_arrays(features.view())
        .expect("the imported model predicts");
    let serialized: serde_json::Map<String, serde_json::Value> = outputs
        .into_iter()
        .map(|(name, array)| {
            let shape: Vec<usize> = array.shape().to_vec();
            let data: Vec<f64> = array.iter().copied().collect();
            (name, serde_json::json!({"shape": shape, "data": data}))
        })
        .collect();
    std::fs::write(
        directory.join("imported.json"),
        serde_json::json!({"outputs": serialized}).to_string(),
    )
    .expect("the imported prediction is writable");
}

fn write_fixture_at(
    directory: &std::path::Path,
    model: &crate::graph::SKOnnxModel,
    features: &Array2<f64>,
    transformed: &Array2<f64>,
) {
    let input = serde_json::json!(
        features
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect::<Vec<Vec<f64>>>()
    );
    let expected = serde_json::json!({
        model.outputs()[0].name(): transformed
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect::<Vec<Vec<f64>>>(),
    });
    model
        .save(directory.join("model.onnx"))
        .expect("the model is writable");
    std::fs::write(
        directory.join("input.json"),
        serde_json::json!({"input_name": "features", "input": input}).to_string(),
    )
    .expect("the input is writable");
    std::fs::write(
        directory.join("expected.json"),
        serde_json::json!({"outputs": expected, "tolerance": 1e-9}).to_string(),
    )
    .expect("the expected output is writable");
}
