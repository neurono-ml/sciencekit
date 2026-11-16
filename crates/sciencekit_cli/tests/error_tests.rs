//! End-to-end error and exit-code contract tests.

use predicates::prelude::*;
use tempfile::tempdir;

mod test_support;

use test_support::{sk_command, sk_dataset};

fn sk_write_workflow(directory: &std::path::Path, body: &str) -> std::path::PathBuf {
    let path = directory.join("workflow.json");
    std::fs::write(&path, body).expect("the workflow file");
    path
}

#[test]
fn invalid_workflow_reports_the_offending_path_with_exit_code_two() {
    let directory = tempdir().expect("a temporary directory");
    let workflow = sk_write_workflow(directory.path(), r#"{"version": 2}"#);
    sk_command()
        .args(["run", workflow.to_str().expect("a UTF-8 path")])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("workflow.version"));

    let workflow = sk_write_workflow(
        directory.path(),
        r#"{"version": 1, "estimator": {"name": "ridge", "typo": 1}}"#,
    );
    sk_command()
        .args(["run", workflow.to_str().expect("a UTF-8 path")])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("workflow.estimator.typo"));
}

#[test]
fn missing_model_is_a_data_error_with_exit_code_three() {
    let directory = tempdir().expect("a temporary directory");
    let dataset = sk_dataset(directory.path());
    let model = directory.path().join("absent.safetensors");
    sk_command()
        .args([
            "predict",
            "--model",
            model.to_str().expect("a UTF-8 path"),
            "--input",
            dataset.to_str().expect("a UTF-8 path"),
            "--output",
            directory
                .path()
                .join("out.csv")
                .to_str()
                .expect("a UTF-8 path"),
        ])
        .assert()
        .code(3)
        .stderr(predicate::str::contains("absent.safetensors"));
}

#[cfg(not(feature = "parquet"))]
#[test]
fn disabled_format_feature_is_an_unsupported_error_with_exit_code_four() {
    let directory = tempdir().expect("a temporary directory");
    let dataset = sk_dataset(directory.path());
    sk_command()
        .args([
            "fit",
            "--input",
            dataset.to_str().expect("a UTF-8 path"),
            "--target",
            "y",
            "--format",
            "parquet",
            "--model",
            directory
                .path()
                .join("model.safetensors")
                .to_str()
                .expect("a UTF-8 path"),
        ])
        .assert()
        .code(4)
        .stderr(predicate::str::contains("`parquet` Cargo feature"));

    let workflow = sk_write_workflow(
        directory.path(),
        r#"{"version": 1, "dataset": {"format": "yaml", "path": "unused", "target": "y"}, "estimator": {"name": "ridge"}}"#,
    );
    sk_command()
        .args(["run", workflow.to_str().expect("a UTF-8 path")])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("configuration.dataset_format"));
}

#[test]
fn usage_failures_exit_with_code_two() {
    sk_command()
        .args(["fit", "--unknown-flag"])
        .assert()
        .code(2);
}

#[cfg(not(feature = "yaml"))]
#[test]
fn a_yaml_workflow_without_the_feature_exits_with_code_four() {
    let directory = tempdir().expect("a temporary directory");
    let workflow = directory.path().join("workflow.yaml");
    std::fs::write(&workflow, "version: 1\n").expect("the YAML workflow");
    sk_command()
        .args(["run", workflow.to_str().expect("a UTF-8 path")])
        .assert()
        .code(4)
        .stderr(predicate::str::contains("`yaml` Cargo feature"));
}
