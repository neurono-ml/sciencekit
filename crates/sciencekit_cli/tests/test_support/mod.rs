#![allow(dead_code)]
//! Shared fixtures and process helpers for the CLI integration suites.

use std::path::{Path, PathBuf};

use assert_cmd::Command;

/// The environment variables the configuration layer reads.
pub const SK_CONFIGURATION_VARIABLES: [&str; 10] = [
    "SCIENCEKIT_DATASET",
    "SCIENCEKIT_DATASET_FORMAT",
    "SCIENCEKIT_TARGET",
    "SCIENCEKIT_ESTIMATOR",
    "SCIENCEKIT_ALPHA",
    "SCIENCEKIT_MODEL",
    "SCIENCEKIT_PREDICTIONS",
    "SCIENCEKIT_METRICS",
    "SCIENCEKIT_OUTPUT_FORMAT",
    "SCIENCEKIT_BACKEND",
];

/// The compiled `sciencekit` binary with every configuration variable cleared.
pub fn sk_command() -> Command {
    let mut command = Command::cargo_bin("sciencekit").expect("the sciencekit binary builds");
    for variable in SK_CONFIGURATION_VARIABLES {
        command.env_remove(variable);
    }
    command
}

/// Write the canonical six-row linear dataset and return its path.
pub fn sk_dataset(directory: &Path) -> PathBuf {
    let path = directory.join("dataset.csv");
    std::fs::write(
        &path,
        "x1,x2,y\n1.0,2.0,3.0\n2.0,4.0,6.0\n3.0,6.0,9.0\n4.0,8.0,12.0\n5.0,10.0,15.0\n6.0,12.0,18.0\n",
    )
    .expect("the fixture dataset");
    path
}

/// A workflow document referencing the canonical dataset.
pub fn sk_workflow(dataset: &Path, directory: &Path) -> String {
    let model = directory.join("workflow-model.safetensors");
    let predictions = directory.join("workflow-predictions.csv");
    format!(
        r#"{{
  "version": 1,
  "dataset": {{ "format": "csv", "path": "{}", "target": "y" }},
  "pipeline": [ {{ "transformer": "standard_scaler", "with_mean": true }} ],
  "estimator": {{ "name": "ridge", "alpha": 1.0 }},
  "search": {{ "kind": "grid", "alphas": [0.1, 1.0], "folds": 2, "scoring": "r2" }},
  "metrics": ["r2", "rmse"],
  "output": {{
    "model": "{}",
    "predictions": "{}",
    "format": "csv"
  }}
}}"#,
        dataset.display(),
        model.display(),
        predictions.display()
    )
}
