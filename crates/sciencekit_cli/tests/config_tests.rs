//! End-to-end configuration-precedence tests (flags > environment > file > defaults).

use serde_json::Value;
use tempfile::tempdir;

mod test_support;

use test_support::{sk_command, sk_dataset};

fn sk_dry_run_alpha(arguments: &[&str], environment: Option<&str>) -> f64 {
    let mut command = sk_command();
    command.args(arguments);
    if let Some(alpha) = environment {
        command.env("SCIENCEKIT_ALPHA", alpha);
    }
    let output = command.output().expect("the binary runs");
    assert!(output.status.success(), "dry run failed: {:?}", output);
    let document: Value = serde_json::from_slice(&output.stdout).expect("the dry run prints JSON");
    document["alpha"].as_f64().expect("an alpha value")
}

#[test]
fn precedence_resolves_flags_over_environment_over_file_over_defaults() {
    let directory = tempdir().expect("a temporary directory");
    let dataset = sk_dataset(directory.path());
    let dataset_text = dataset.to_str().expect("a UTF-8 path");
    let config = directory.path().join("config.json");
    std::fs::write(
        &config,
        format!(
            r#"{{"version": 1, "dataset": {{"path": "{dataset_text}", "target": "y"}}, "estimator": {{"name": "ridge", "alpha": 2.0}}}}"#
        ),
    )
    .expect("the config file");
    let config_text = config.to_str().expect("a UTF-8 path");

    assert_eq!(sk_dry_run_alpha(&["fit", "--dry-run"], None), 1.0);
    assert_eq!(
        sk_dry_run_alpha(&["fit", "--dry-run", "--config", config_text], None),
        2.0
    );
    assert_eq!(
        sk_dry_run_alpha(&["fit", "--dry-run", "--config", config_text], Some("3.0")),
        3.0
    );
    assert_eq!(
        sk_dry_run_alpha(
            &[
                "fit",
                "--dry-run",
                "--config",
                config_text,
                "--alpha",
                "4.0",
            ],
            Some("3.0")
        ),
        4.0
    );
    assert_eq!(
        sk_dry_run_alpha(&["fit", "--dry-run", "--input", dataset_text], None),
        1.0
    );
}

#[test]
fn the_workflow_file_output_format_beats_extension_inference() {
    let directory = tempdir().expect("a temporary directory");
    let dataset = sk_dataset(directory.path());
    let model = directory.path().join("model.safetensors");
    let config = directory.path().join("config.json");
    std::fs::write(
        &config,
        format!(
            r#"{{"version": 1, "dataset": {{"path": "{}", "target": "y"}}, "estimator": {{"name": "ridge"}}, "output": {{"format": "json"}}}}"#,
            dataset.display()
        ),
    )
    .expect("the config file");
    let config_text = config.to_str().expect("a UTF-8 path");

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

    let predictions = directory.path().join("predictions.csv");
    sk_command()
        .args([
            "predict",
            "--config",
            config_text,
            "--model",
            model.to_str().expect("a UTF-8 path"),
            "--input",
            dataset.to_str().expect("a UTF-8 path"),
            "--output",
            predictions.to_str().expect("a UTF-8 path"),
        ])
        .assert()
        .success();
    let text = std::fs::read_to_string(&predictions).expect("the predictions file");
    assert!(
        text.starts_with("[["),
        "the workflow file's `output.format` must win over the `.csv` extension, got {text}"
    );

    let scaler = directory.path().join("scaler.safetensors");
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

    let transformed = directory.path().join("transformed.csv");
    sk_command()
        .args([
            "transform",
            "--config",
            config_text,
            "--model",
            scaler.to_str().expect("a UTF-8 path"),
            "--input",
            dataset.to_str().expect("a UTF-8 path"),
            "--output",
            transformed.to_str().expect("a UTF-8 path"),
        ])
        .assert()
        .success();
    let text = std::fs::read_to_string(&transformed).expect("the transformed matrix");
    assert!(
        text.starts_with("[["),
        "the workflow file's `output.format` must win over the `.csv` extension, got {text}"
    );
}

#[test]
fn an_empty_output_format_variable_leaves_the_extension_in_charge() {
    let directory = tempdir().expect("a temporary directory");
    let dataset = sk_dataset(directory.path());
    let model = directory.path().join("model.safetensors");
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

    let predictions = directory.path().join("predictions.json");
    let mut command = sk_command();
    command
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
        .env("SCIENCEKIT_OUTPUT_FORMAT", "")
        .assert()
        .success();
    let text = std::fs::read_to_string(&predictions).expect("the predictions file");
    assert!(
        text.starts_with("[["),
        "an empty environment value must not force CSV, got {text}"
    );
}
