//! End-to-end command tests: each one runs the compiled binary.

use predicates::prelude::*;
use tempfile::tempdir;

mod test_support;

use test_support::{sk_command, sk_dataset};

#[test]
fn fit_then_predict_inspect_and_export_run_end_to_end() {
    let directory = tempdir().expect("a temporary directory");
    let dataset = sk_dataset(directory.path());
    let model = directory.path().join("model.safetensors");
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
            "--estimator",
            "ridge",
            "--alpha",
            "1.0",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"command\": \"fit\""));
    assert!(model.exists(), "fit writes the model artifact");

    sk_command()
        .args([
            "predict",
            "--model",
            model.to_str().expect("a UTF-8 path"),
            "--input",
            dataset.to_str().expect("a UTF-8 path"),
            "--target",
            "y",
            "--output",
            predictions.to_str().expect("a UTF-8 path"),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"command\": \"predict\""));
    let text = std::fs::read_to_string(&predictions).expect("the predictions file");
    assert_eq!(text.lines().count(), 7, "header plus six rows");
    let first = text.lines().nth(1).expect("the first prediction");
    let value: f64 = first.trim().parse().expect("a numeric prediction");
    assert!((value - 3.0).abs() < 0.2, "prediction was {value}");

    sk_command()
        .args(["inspect", "--model", model.to_str().expect("a UTF-8 path")])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"estimator\": \"SKRidge\""))
        .stdout(predicate::str::contains("\"tensor_count\""));

    let exported = directory.path().join("model.json");
    sk_command()
        .args([
            "export",
            "--model",
            model.to_str().expect("a UTF-8 path"),
            "--output",
            exported.to_str().expect("a UTF-8 path"),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"format\": \"json\""));
    sk_command()
        .args([
            "inspect",
            "--model",
            exported.to_str().expect("a UTF-8 path"),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"format\": \"json\""));
}

#[test]
fn transform_evaluate_and_run_workflow_produce_artifacts() {
    let directory = tempdir().expect("a temporary directory");
    let dataset = sk_dataset(directory.path());
    let scaler = directory.path().join("scaler.safetensors");
    let transformed = directory.path().join("transformed.csv");

    sk_command()
        .args([
            "fit",
            "--input",
            dataset.to_str().expect("a UTF-8 path"),
            "--target",
            "y",
            "--estimator",
            "standard_scaler",
            "--model",
            scaler.to_str().expect("a UTF-8 path"),
        ])
        .assert()
        .success();
    sk_command()
        .args([
            "transform",
            "--model",
            scaler.to_str().expect("a UTF-8 path"),
            "--input",
            dataset.to_str().expect("a UTF-8 path"),
            "--target",
            "y",
            "--output",
            transformed.to_str().expect("a UTF-8 path"),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"number_of_features\": 2"));
    let text = std::fs::read_to_string(&transformed).expect("the transformed matrix");
    assert_eq!(text.lines().count(), 7);

    let model = directory.path().join("evaluated.safetensors");
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
            "evaluate",
            "--model",
            model.to_str().expect("a UTF-8 path"),
            "--input",
            dataset.to_str().expect("a UTF-8 path"),
            "--target",
            "y",
            "--metrics",
            "r2,rmse",
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"r2\""))
        .stdout(predicate::str::contains("\"rmse\""));

    let workflow = directory.path().join("workflow.json");
    std::fs::write(
        &workflow,
        test_support::sk_workflow(&dataset, directory.path()),
    )
    .expect("the workflow file");
    sk_command()
        .args(["run", workflow.to_str().expect("a UTF-8 path")])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"command\": \"run\""))
        .stdout(predicate::str::contains("\"best_alpha\""));
    assert!(directory.path().join("workflow-model.safetensors").exists());
    assert!(directory.path().join("workflow-predictions.csv").exists());
}
