//! Structural and semantic validation of a parsed workflow document.

use serde_json::{Map, Value};

use crate::workflow::error::SKWorkflowError;
use crate::workflow::schema::{
    SK_WORKFLOW_SUPPORTED_VERSION, SK_WORKFLOW_VERSION_PATH, SKWorkflow,
};

const SK_WORKFLOW_KEYS: [&str; 7] = [
    "version",
    "dataset",
    "pipeline",
    "estimator",
    "search",
    "metrics",
    "output",
];
const SK_DATASET_KEYS: [&str; 3] = ["format", "path", "target"];
const SK_PIPELINE_STEP_KEYS: [&str; 3] = ["transformer", "with_mean", "with_std"];
const SK_ESTIMATOR_KEYS: [&str; 3] = ["name", "alpha", "backend"];
const SK_SEARCH_KEYS: [&str; 4] = ["kind", "alphas", "folds", "scoring"];
const SK_OUTPUT_KEYS: [&str; 3] = ["model", "predictions", "format"];
const SK_SUPPORTED_TRANSFORMERS: [&str; 3] = ["standard_scaler", "minmax_scaler", "robust_scaler"];
const SK_SUPPORTED_ESTIMATORS: [&str; 2] = ["ridge", "linear_regression"];
const SK_SUPPORTED_METRICS: [&str; 4] = ["r2", "rmse", "mse", "mae"];

fn sk_object<'a>(value: &'a Value, path: &str) -> Result<&'a Map<String, Value>, SKWorkflowError> {
    value
        .as_object()
        .ok_or_else(|| SKWorkflowError::InvalidType {
            path: path.to_string(),
            expected: "JSON object",
        })
}

fn sk_check_keys(
    object: &Map<String, Value>,
    path: &str,
    expected: &[&'static str],
) -> Result<(), SKWorkflowError> {
    for key in object.keys() {
        if !expected.contains(&key.as_str()) {
            return Err(SKWorkflowError::UnknownField {
                path: format!("{path}.{key}"),
                field: key.clone(),
                expected: expected.to_vec(),
            });
        }
    }
    Ok(())
}

fn sk_optional_string(
    object: &Map<String, Value>,
    key: &str,
    path: &str,
) -> Result<(), SKWorkflowError> {
    match object.get(key) {
        None => Ok(()),
        Some(Value::String(_)) => Ok(()),
        Some(_) => Err(SKWorkflowError::InvalidType {
            path: format!("{path}.{key}"),
            expected: "string",
        }),
    }
}

/// Validate the document structure and version, naming the offending path.
pub fn sk_validate_workflow_value(root: &Value) -> Result<(), SKWorkflowError> {
    let object = sk_object(root, "workflow")?;
    sk_check_keys(object, "workflow", &SK_WORKFLOW_KEYS)?;
    let version = object
        .get("version")
        .ok_or_else(|| SKWorkflowError::MissingField {
            path: SK_WORKFLOW_VERSION_PATH.to_string(),
        })?
        .as_u64()
        .ok_or_else(|| SKWorkflowError::InvalidType {
            path: SK_WORKFLOW_VERSION_PATH.to_string(),
            expected: "non-negative integer",
        })?;
    if version != u64::from(SK_WORKFLOW_SUPPORTED_VERSION) {
        return Err(SKWorkflowError::UnsupportedVersion {
            path: SK_WORKFLOW_VERSION_PATH.to_string(),
            found: version,
            supported: SK_WORKFLOW_SUPPORTED_VERSION,
        });
    }

    if let Some(dataset) = object.get("dataset") {
        let dataset = sk_object(dataset, "workflow.dataset")?;
        sk_check_keys(dataset, "workflow.dataset", &SK_DATASET_KEYS)?;
        if !matches!(dataset.get("path"), Some(Value::String(_))) {
            return Err(SKWorkflowError::MissingField {
                path: "workflow.dataset.path".to_string(),
            });
        }
        sk_optional_string(dataset, "format", "workflow.dataset")?;
        sk_optional_string(dataset, "target", "workflow.dataset")?;
    }

    if let Some(pipeline) = object.get("pipeline") {
        let steps = pipeline
            .as_array()
            .ok_or_else(|| SKWorkflowError::InvalidType {
                path: "workflow.pipeline".to_string(),
                expected: "array",
            })?;
        for (position, step) in steps.iter().enumerate() {
            let path = format!("workflow.pipeline[{position}]");
            let step = sk_object(step, &path)?;
            sk_check_keys(step, &path, &SK_PIPELINE_STEP_KEYS)?;
            if !matches!(step.get("transformer"), Some(Value::String(_))) {
                return Err(SKWorkflowError::MissingField {
                    path: format!("{path}.transformer"),
                });
            }
        }
    }

    if let Some(estimator) = object.get("estimator") {
        let estimator = sk_object(estimator, "workflow.estimator")?;
        sk_check_keys(estimator, "workflow.estimator", &SK_ESTIMATOR_KEYS)?;
        if !matches!(estimator.get("name"), Some(Value::String(_))) {
            return Err(SKWorkflowError::MissingField {
                path: "workflow.estimator.name".to_string(),
            });
        }
        sk_optional_string(estimator, "backend", "workflow.estimator")?;
    }

    if let Some(search) = object.get("search") {
        let search = sk_object(search, "workflow.search")?;
        sk_check_keys(search, "workflow.search", &SK_SEARCH_KEYS)?;
        if !matches!(search.get("kind"), Some(Value::String(_))) {
            return Err(SKWorkflowError::MissingField {
                path: "workflow.search.kind".to_string(),
            });
        }
    }

    if let Some(output) = object.get("output") {
        let output = sk_object(output, "workflow.output")?;
        sk_check_keys(output, "workflow.output", &SK_OUTPUT_KEYS)?;
        sk_optional_string(output, "model", "workflow.output")?;
        sk_optional_string(output, "predictions", "workflow.output")?;
        sk_optional_string(output, "format", "workflow.output")?;
    }

    Ok(())
}

/// Validate the semantic constraints a well-formed document can still violate.
pub fn sk_validate_workflow_semantics(workflow: &SKWorkflow) -> Result<(), SKWorkflowError> {
    let dataset = workflow
        .dataset
        .as_ref()
        .ok_or_else(|| SKWorkflowError::MissingField {
            path: "workflow.dataset".to_string(),
        })?;
    if dataset.path.trim().is_empty() {
        return Err(SKWorkflowError::MissingField {
            path: "workflow.dataset.path".to_string(),
        });
    }
    let estimator = workflow
        .estimator
        .as_ref()
        .ok_or_else(|| SKWorkflowError::MissingField {
            path: "workflow.estimator".to_string(),
        })?;
    if !SK_SUPPORTED_ESTIMATORS.contains(&estimator.name.as_str()) {
        return Err(SKWorkflowError::UnknownField {
            path: "workflow.estimator.name".to_string(),
            field: estimator.name.clone(),
            expected: SK_SUPPORTED_ESTIMATORS.to_vec(),
        });
    }
    if let Some(alpha) = estimator.alpha {
        if !alpha.is_finite() || alpha <= 0.0 {
            return Err(SKWorkflowError::InvalidType {
                path: "workflow.estimator.alpha".to_string(),
                expected: "finite positive number",
            });
        }
    }
    for (position, step) in workflow.pipeline.iter().enumerate() {
        if !SK_SUPPORTED_TRANSFORMERS.contains(&step.transformer.as_str()) {
            return Err(SKWorkflowError::UnknownField {
                path: format!("workflow.pipeline[{position}].transformer"),
                field: step.transformer.clone(),
                expected: SK_SUPPORTED_TRANSFORMERS.to_vec(),
            });
        }
    }
    if let Some(search) = &workflow.search {
        if search.kind != "grid" {
            return Err(SKWorkflowError::UnknownField {
                path: "workflow.search.kind".to_string(),
                field: search.kind.clone(),
                expected: vec!["grid"],
            });
        }
        if search.alphas.is_empty() {
            return Err(SKWorkflowError::MissingField {
                path: "workflow.search.alphas".to_string(),
            });
        }
        if search
            .alphas
            .iter()
            .any(|alpha| !alpha.is_finite() || *alpha <= 0.0)
        {
            return Err(SKWorkflowError::InvalidType {
                path: "workflow.search.alphas".to_string(),
                expected: "array of finite positive numbers",
            });
        }
        if search.folds.is_some_and(|folds| folds < 2) {
            return Err(SKWorkflowError::InvalidType {
                path: "workflow.search.folds".to_string(),
                expected: "integer of at least 2",
            });
        }
    }
    for (position, metric) in workflow.metrics.iter().enumerate() {
        if !SK_SUPPORTED_METRICS.contains(&metric.as_str()) {
            return Err(SKWorkflowError::UnknownField {
                path: format!("workflow.metrics[{position}]"),
                field: metric.clone(),
                expected: SK_SUPPORTED_METRICS.to_vec(),
            });
        }
    }
    Ok(())
}
