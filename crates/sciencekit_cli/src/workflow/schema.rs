//! The versioned workflow schema (spec `workflows-cli`, requirement "Workflow
//! schema").

use serde::{Deserialize, Serialize};

/// The only workflow major version this build executes.
pub const SK_WORKFLOW_SUPPORTED_VERSION: u32 = 1;

/// The dotted path of the version field.
pub const SK_WORKFLOW_VERSION_PATH: &str = "workflow.version";

/// A reproducible experiment record: dataset, pipeline, estimator, search,
/// metrics and output artifacts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SKWorkflow {
    /// The schema major version.
    pub version: u32,
    /// The dataset source.
    #[serde(default)]
    pub dataset: Option<SKWorkflowDataset>,
    /// Ordered feature transformers applied before the estimator.
    #[serde(default)]
    pub pipeline: Vec<SKWorkflowPipelineStep>,
    /// The estimator configuration.
    #[serde(default)]
    pub estimator: Option<SKWorkflowEstimator>,
    /// Optional grid search over penalty values.
    #[serde(default)]
    pub search: Option<SKWorkflowSearch>,
    /// Metric names computed on the fitted model.
    #[serde(default)]
    pub metrics: Vec<String>,
    /// Declared output artifacts.
    #[serde(default)]
    pub output: Option<SKWorkflowOutput>,
}

/// The dataset source of a workflow.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SKWorkflowDataset {
    /// `csv` or `parquet`; inferred from the extension when absent.
    #[serde(default)]
    pub format: Option<String>,
    /// The file path.
    pub path: String,
    /// The target column name.
    #[serde(default)]
    pub target: Option<String>,
}

/// One transformer step in the feature pipeline.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SKWorkflowPipelineStep {
    /// The transformer name: `standard_scaler`, `minmax_scaler` or
    /// `robust_scaler`.
    pub transformer: String,
    /// The standard scaler's `with_mean` switch.
    #[serde(default)]
    pub with_mean: Option<bool>,
    /// The standard scaler's `with_std` switch.
    #[serde(default)]
    pub with_std: Option<bool>,
}

/// The estimator configuration.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SKWorkflowEstimator {
    /// The estimator name: `ridge`, `linear_regression` or a scaler.
    pub name: String,
    /// The ridge penalty.
    #[serde(default)]
    pub alpha: Option<f64>,
    /// The math backend kind.
    #[serde(default)]
    pub backend: Option<String>,
}

/// A grid search over ridge penalties.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct SKWorkflowSearch {
    /// The only supported kind, `grid`.
    pub kind: String,
    /// The penalty candidates.
    #[serde(default)]
    pub alphas: Vec<f64>,
    /// The number of cross-validation folds.
    #[serde(default)]
    pub folds: Option<usize>,
    /// The scoring name, `r2` by default.
    #[serde(default)]
    pub scoring: Option<String>,
}

/// The declared output artifacts.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct SKWorkflowOutput {
    /// The model artifact path.
    #[serde(default)]
    pub model: Option<String>,
    /// The predictions artifact path.
    #[serde(default)]
    pub predictions: Option<String>,
    /// The predictions or matrix format.
    #[serde(default)]
    pub format: Option<String>,
}
