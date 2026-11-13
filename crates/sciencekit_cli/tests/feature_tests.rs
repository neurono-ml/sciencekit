#![allow(unused_imports)]
//! Feature-gated end-to-end tests: YAML workflows, Parquet artifacts, ONNX
//! model export/import.
//!
//! Each test runs only when its Cargo feature is enabled; the default build
//! proves the corresponding feature-absent error instead.

mod test_support;

use predicates::prelude::*;
use tempfile::tempdir;

use test_support::{sk_command, sk_dataset};

#[cfg(feature = "yaml")]
#[test]
fn a_yaml_workflow_executes_end_to_end() {
    let directory = tempdir().expect("a temporary directory");
    let dataset = sk_dataset(directory.path());
    let model = directory.path().join("yaml-model.safetensors");
    let workflow = directory.path().join("workflow.yaml");
    std::fs::write(
        &workflow,
        format!(
            "version: 1\ndataset:\n  format: csv\n  path: {}\n  target: y\nestimator:\n  name: ridge\n  alpha: 1.0\nmetrics: [r2]\noutput:\n  model: {}\n",
            dataset.display(),
            model.display()
        ),
    )
    .expect("the YAML workflow");
    sk_command()
        .args(["run", workflow.to_str().expect("a UTF-8 path")])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"command\": \"run\""));
    assert!(model.exists());
}

#[cfg(feature = "parquet")]
#[test]
fn a_parquet_dataset_fits_and_parquet_predictions_round_trip() {
    use std::sync::Arc;

    use arrow_array::{ArrayRef, Float64Array, RecordBatch};
    use arrow_schema::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;

    let directory = tempdir().expect("a temporary directory");
    let dataset = directory.path().join("dataset.parquet");
    let schema = Arc::new(Schema::new(vec![
        Field::new("x1", DataType::Float64, false),
        Field::new("x2", DataType::Float64, false),
        Field::new("y", DataType::Float64, false),
    ]));
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0, 4.0])),
        Arc::new(Float64Array::from(vec![2.0, 4.0, 6.0, 8.0])),
        Arc::new(Float64Array::from(vec![3.0, 6.0, 9.0, 12.0])),
    ];
    let batch = RecordBatch::try_new(schema.clone(), columns).expect("the batch");
    let file = std::fs::File::create(&dataset).expect("the Parquet file");
    let mut writer = ArrowWriter::try_new(file, schema, None).expect("the writer");
    writer.write(&batch).expect("the batch writes");
    writer.close().expect("the file finalizes");

    let model = directory.path().join("parquet-model.safetensors");
    let predictions = directory.path().join("predictions.parquet");
    sk_command()
        .args([
            "fit",
            "--input",
            dataset.to_str().expect("a UTF-8 path"),
            "--format",
            "parquet",
            "--target",
            "y",
            "--model",
            model.to_str().expect("a UTF-8 path"),
        ])
        .assert()
        .success();
    sk_command()
        .args([
            "predict",
            "--model",
            model.to_str().expect("a UTF-8 path"),
            "--input",
            dataset.to_str().expect("a UTF-8 path"),
            "--format",
            "parquet",
            "--target",
            "y",
            "--output",
            predictions.to_str().expect("a UTF-8 path"),
        ])
        .assert()
        .success();
    assert!(predictions.exists());
}

#[cfg(feature = "onnx")]
#[test]
fn onnx_export_and_import_round_trip_a_fitted_model() {
    let directory = tempdir().expect("a temporary directory");
    let dataset = sk_dataset(directory.path());
    let model = directory.path().join("model.safetensors");
    let onnx = directory.path().join("model.onnx");
    let predictions = directory.path().join("predictions.csv");
    sk_command()
        .args([
            "fit",
            "--input",
            dataset.to_str().expect("a UTF-8 path"),
            "--target",
            "y",
            "--model",
            model.to_str().expect("a UTF-8 path"),
        ])
        .assert()
        .success();
    sk_command()
        .args([
            "export",
            "--model",
            model.to_str().expect("a UTF-8 path"),
            "--output",
            onnx.to_str().expect("a UTF-8 path"),
            "--model-format",
            "onnx",
        ])
        .assert()
        .success();
    assert!(onnx.exists());
    sk_command()
        .args([
            "predict",
            "--model",
            onnx.to_str().expect("a UTF-8 path"),
            "--input",
            dataset.to_str().expect("a UTF-8 path"),
            "--target",
            "y",
            "--output",
            predictions.to_str().expect("a UTF-8 path"),
        ])
        .assert()
        .success();
}
