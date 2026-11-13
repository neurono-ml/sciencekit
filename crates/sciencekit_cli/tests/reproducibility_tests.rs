//! Reproducibility of workflow artifacts: two runs of one workflow must write
//! byte-identical files (spec `workflows-cli`, proposal "reproducible
//! experiment record").

use tempfile::tempdir;

mod test_support;

use test_support::{sk_command, sk_dataset};

fn sk_workflow_for_directory(directory: &std::path::Path, dataset: &std::path::Path) -> String {
    let model = directory.join("model.safetensors");
    let predictions = directory.join("predictions.csv");
    format!(
        r#"{{
  "version": 1,
  "dataset": {{ "format": "csv", "path": "{}", "target": "y" }},
  "pipeline": [ {{ "transformer": "standard_scaler" }} ],
  "estimator": {{ "name": "ridge", "alpha": 1.0 }},
  "search": {{ "kind": "grid", "alphas": [0.1, 1.0], "folds": 2 }},
  "metrics": ["r2", "rmse"],
  "output": {{ "model": "{}", "predictions": "{}", "format": "csv" }}
}}"#,
        dataset.display(),
        model.display(),
        predictions.display()
    )
}

#[test]
fn the_same_workflow_writes_byte_identical_artifacts_into_two_directories() {
    let directory = tempdir().expect("a temporary directory");
    let dataset = sk_dataset(directory.path());
    let first = directory.path().join("first");
    let second = directory.path().join("second");
    std::fs::create_dir(&first).expect("the first output directory");
    std::fs::create_dir(&second).expect("the second output directory");

    for (name, output) in [("first.json", &first), ("second.json", &second)] {
        let workflow = directory.path().join(name);
        std::fs::write(&workflow, sk_workflow_for_directory(output, &dataset))
            .expect("the workflow file");
        sk_command()
            .args(["run", workflow.to_str().expect("a UTF-8 path")])
            .assert()
            .success();
    }

    for artifact in ["model.safetensors", "predictions.csv"] {
        let first_bytes = std::fs::read(first.join(artifact)).expect("the first artifact");
        let second_bytes = std::fs::read(second.join(artifact)).expect("the second artifact");
        assert_eq!(
            first_bytes, second_bytes,
            "`{artifact}` must be byte-identical across runs"
        );
    }
}
