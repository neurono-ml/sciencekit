//! A workflow that cannot write every declared artifact must fail before it
//! writes any of them: a failed run leaves no partial experiment record.

use predicates::prelude::*;
use tempfile::tempdir;

mod test_support;

use test_support::{sk_command, sk_dataset};

#[test]
fn an_unwritable_predictions_path_writes_no_model_artifact() {
    let directory = tempdir().expect("a temporary directory");
    let dataset = sk_dataset(directory.path());
    let model = directory.path().join("model.safetensors");
    let workflow = directory.path().join("workflow.json");
    std::fs::write(
        &workflow,
        format!(
            r#"{{"version": 1, "dataset": {{"path": "{}", "target": "y"}}, "estimator": {{"name": "ridge"}}, "metrics": ["r2"], "output": {{"model": "{}", "predictions": "{}", "format": "csv"}}}}"#,
            dataset.display(),
            model.display(),
            directory.path().join("absent-directory/predictions.csv").display()
        ),
    )
    .expect("the workflow file");

    sk_command()
        .args(["run", workflow.to_str().expect("a UTF-8 path")])
        .assert()
        .code(2)
        .stderr(predicate::str::contains("configuration.predictions"));
    assert!(
        !model.exists(),
        "the model must not be written when the predictions path is unusable"
    );
}

#[cfg(not(feature = "parquet"))]
#[test]
fn an_unsupported_predictions_format_writes_no_model_artifact() {
    let directory = tempdir().expect("a temporary directory");
    let dataset = sk_dataset(directory.path());
    let model = directory.path().join("model.safetensors");
    let workflow = directory.path().join("workflow.json");
    std::fs::write(
        &workflow,
        format!(
            r#"{{"version": 1, "dataset": {{"path": "{}", "target": "y"}}, "estimator": {{"name": "ridge"}}, "metrics": ["r2"], "output": {{"model": "{}", "predictions": "{}", "format": "parquet"}}}}"#,
            dataset.display(),
            model.display(),
            directory.path().join("predictions.parquet").display()
        ),
    )
    .expect("the workflow file");

    sk_command()
        .args(["run", workflow.to_str().expect("a UTF-8 path")])
        .assert()
        .code(4)
        .stderr(predicate::str::contains("`parquet` Cargo feature"));
    assert!(
        !model.exists(),
        "the model must not be written when the predictions format is unavailable"
    );
}
