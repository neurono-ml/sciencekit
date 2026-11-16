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
