//! Unit tests for the workflow schema, versioning and unknown-field validation.

use serde_json::json;

use crate::workflow::error::SKWorkflowError;
use crate::workflow::loader::{SKWorkflowFormat, sk_parse_workflow};
use crate::workflow::schema::SK_WORKFLOW_SUPPORTED_VERSION;
use crate::workflow::validation::sk_validate_workflow_value;

#[test]
fn a_supported_version_parses_into_the_schema() {
    let workflow = sk_parse_workflow(
        r#"{
          "version": 1,
          "dataset": {"format": "csv", "path": "data.csv", "target": "y"},
          "pipeline": [{"transformer": "standard_scaler", "with_mean": true}],
          "estimator": {"name": "ridge", "alpha": 1.0},
          "search": {"kind": "grid", "alphas": [0.1, 1.0], "folds": 2},
          "metrics": ["r2", "rmse"],
          "output": {"model": "model.safetensors"}
        }"#,
        SKWorkflowFormat::Json,
    )
    .expect("the workflow parses");
    assert_eq!(workflow.version, SK_WORKFLOW_SUPPORTED_VERSION);
    assert_eq!(workflow.pipeline.len(), 1);
    assert_eq!(workflow.search.expect("a search").alphas.len(), 2);
}

#[test]
fn an_unknown_major_version_is_a_typed_error_naming_the_path() {
    let error =
        sk_validate_workflow_value(&json!({"version": 2})).expect_err("version 2 is not supported");
    assert_eq!(
        error,
        SKWorkflowError::UnsupportedVersion {
            path: "workflow.version".to_string(),
            found: 2,
            supported: 1,
        }
    );
    assert_eq!(error.path(), "workflow.version");
}

#[test]
fn an_unknown_nested_field_is_rejected_with_its_dotted_path() {
    let error = sk_validate_workflow_value(&json!({
        "version": 1,
        "search": {"kind": "grid", "alphas": [1.0], "typo": true}
    }))
    .expect_err("the unknown field is rejected");
    assert!(matches!(
        &error,
        SKWorkflowError::UnknownField { path, field, .. }
            if path == "workflow.search.typo" && field == "typo"
    ));
}

#[test]
fn toml_workflows_parse_through_the_same_schema() {
    let workflow = sk_parse_workflow(
        r#"
version = 1
metrics = ["r2"]
[dataset]
format = "csv"
path = "data.csv"
target = "y"
[estimator]
name = "ridge"
alpha = 0.5
"#,
        SKWorkflowFormat::Toml,
    )
    .expect("the TOML workflow parses");
    assert_eq!(workflow.estimator.expect("an estimator").alpha, Some(0.5));
}

#[cfg(feature = "yaml")]
#[test]
fn yaml_workflows_parse_when_the_feature_is_enabled() {
    let workflow = sk_parse_workflow(
        "version: 1\ndataset:\n  path: data.csv\n  target: y\nestimator:\n  name: ridge\n",
        SKWorkflowFormat::Yaml,
    )
    .expect("the YAML workflow parses");
    assert_eq!(workflow.version, 1);
}
