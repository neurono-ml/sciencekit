//! Workflow file loading for JSON, TOML and the optional YAML feature.

use std::path::Path;

use serde_json::Value;

use crate::error::SKCliError;
use crate::workflow::schema::SKWorkflow;
use crate::workflow::validation::{sk_validate_workflow_semantics, sk_validate_workflow_value};

/// The workflow file formats the loader recognizes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKWorkflowFormat {
    /// JSON.
    Json,
    /// TOML.
    Toml,
    /// YAML behind the `yaml` feature.
    Yaml,
}

impl SKWorkflowFormat {
    /// Infer the format from the path extension, falling back to JSON.
    pub fn from_path(path: &Path) -> Self {
        match path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("toml") => SKWorkflowFormat::Toml,
            Some("yaml" | "yml") => SKWorkflowFormat::Yaml,
            _ => SKWorkflowFormat::Json,
        }
    }
}

fn sk_parse_document(text: &str, format: SKWorkflowFormat) -> Result<Value, SKCliError> {
    match format {
        SKWorkflowFormat::Json => serde_json::from_str(text)
            .map_err(|error| SKCliError::validation("workflow", error.to_string())),
        SKWorkflowFormat::Toml => {
            let value: toml::Value = toml::from_str(text)
                .map_err(|error| SKCliError::validation("workflow", error.to_string()))?;
            serde_json::to_value(value)
                .map_err(|error| SKCliError::validation("workflow", error.to_string()))
        }
        SKWorkflowFormat::Yaml => {
            #[cfg(feature = "yaml")]
            {
                let value: serde_yaml_ng::Value = serde_yaml_ng::from_str(text)
                    .map_err(|error| SKCliError::validation("workflow", error.to_string()))?;
                serde_json::to_value(value)
                    .map_err(|error| SKCliError::validation("workflow", error.to_string()))
            }
            #[cfg(not(feature = "yaml"))]
            {
                let _ = text;
                Err(SKCliError::unsupported("yaml", "a YAML workflow file"))
            }
        }
    }
}

/// Parse a workflow document from text.
pub fn sk_parse_workflow(text: &str, format: SKWorkflowFormat) -> Result<SKWorkflow, SKCliError> {
    let value = sk_parse_document(text, format)?;
    sk_validate_workflow_value(&value).map_err(|error| error.into_cli_error())?;
    let workflow: SKWorkflow = serde_json::from_value(value)
        .map_err(|error| SKCliError::validation("workflow", error.to_string()))?;
    sk_validate_workflow_semantics(&workflow).map_err(|error| error.into_cli_error())?;
    Ok(workflow)
}

/// Read and validate a workflow file from disk.
pub fn sk_load_workflow(path: &Path) -> Result<SKWorkflow, SKCliError> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| SKCliError::data(path.display().to_string(), error.to_string()))?;
    sk_parse_workflow(&text, SKWorkflowFormat::from_path(path))
}
