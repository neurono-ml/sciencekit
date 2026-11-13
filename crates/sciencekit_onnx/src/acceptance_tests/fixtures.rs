//! Fixture files consumed by `validation/compare_with_onnxruntime.py`.
//!
//! A test writes a fixture only when `SCIENCEKIT_ONNX_FIXTURE_DIR` is set, so a
//! normal `cargo test` stays pure while the validation run produces the exact
//! bytes the reference runtime re-executes.

use std::path::PathBuf;

use serde_json::{Value, json};

use crate::graph::SKOnnxModel;

/// The directory the validation run collects fixtures under, when requested.
pub(crate) fn fixture_directory() -> Option<PathBuf> {
    std::env::var_os("SCIENCEKIT_ONNX_FIXTURE_DIR").map(PathBuf::from)
}

/// Write one model, its input and the Rust prediction for the reference runtime.
pub(crate) fn write_fixture(
    fixture_name: &str,
    model: &SKOnnxModel,
    input_name: &str,
    input: Value,
    outputs: Value,
    tolerance: f64,
) {
    let Some(root) = fixture_directory() else {
        return;
    };
    let directory = root.join(fixture_name);
    std::fs::create_dir_all(&directory).expect("the fixture directory is writable");
    model
        .save(directory.join("model.onnx"))
        .expect("the model is writable");
    std::fs::write(
        directory.join("input.json"),
        json!({"input_name": input_name, "input": input}).to_string(),
    )
    .expect("the input is writable");
    std::fs::write(
        directory.join("expected.json"),
        json!({"outputs": outputs, "tolerance": tolerance}).to_string(),
    )
    .expect("the expected output is writable");
}
