//! Dry-run must reject every value the command would reject on execution: a
//! resolved configuration that prints cleanly must be executable.

use predicates::prelude::*;
use tempfile::tempdir;

mod test_support;

use test_support::{sk_command, sk_dataset};

fn sk_assert_dry_run(arguments: &[&str], code: i32, message: &str) {
    sk_command()
        .args(arguments)
        .assert()
        .code(code)
        .stderr(predicate::str::contains(message));
}

#[test]
fn dry_run_rejects_every_invalid_configuration_value() {
    sk_assert_dry_run(
        &["fit", "--dry-run", "--format", "arff"],
        2,
        "configuration.dataset_format",
    );
    sk_assert_dry_run(
        &["fit", "--dry-run", "--output-format", "bogus"],
        2,
        "configuration.output_format",
    );
    sk_assert_dry_run(
        &["fit", "--dry-run", "--metrics", "r2,bogus"],
        2,
        "configuration.metrics",
    );
    sk_assert_dry_run(
        &["fit", "--dry-run", "--estimator", "gradient_boosting"],
        2,
        "configuration.estimator",
    );
    sk_assert_dry_run(
        &["fit", "--dry-run", "--backend", "opencl"],
        4,
        "`opencl` Cargo feature",
    );
}

#[test]
fn dry_run_rejects_an_invalid_workflow_output_format() {
    let directory = tempdir().expect("a temporary directory");
    let dataset = sk_dataset(directory.path());
    let workflow = directory.path().join("workflow.json");
    std::fs::write(
        &workflow,
        format!(
            r#"{{"version": 1, "dataset": {{"path": "{}", "target": "y"}}, "estimator": {{"name": "ridge"}}, "output": {{"format": "bogus"}}}}"#,
            dataset.display()
        ),
    )
    .expect("the workflow file");
    sk_assert_dry_run(
        &["run", workflow.to_str().expect("a UTF-8 path"), "--dry-run"],
        2,
        "configuration.output_format",
    );
}

#[test]
fn dry_run_rejects_invalid_command_specific_formats() {
    let directory = tempdir().expect("a temporary directory");
    let model = directory.path().join("model.safetensors");
    sk_command()
        .args([
            "export",
            "--dry-run",
            "--model",
            model.to_str().expect("a UTF-8 path"),
            "--output",
            "model.bin",
            "--model-format",
            "bogus",
        ])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("configuration.model_format"));
}
