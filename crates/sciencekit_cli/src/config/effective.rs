//! The resolved configuration shared by every command.

use serde::Serialize;

/// The effective configuration after defaults, file, environment and flags
/// were merged in that order.
#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct SKEffectiveConfiguration {
    /// The dataset path.
    pub dataset: Option<String>,
    /// The dataset format (`csv` or `parquet`).
    pub dataset_format: Option<String>,
    /// The target column name.
    pub target: Option<String>,
    /// The estimator name.
    pub estimator: String,
    /// The ridge penalty.
    pub alpha: f64,
    /// The model artifact path.
    pub model: Option<String>,
    /// The predictions artifact path.
    pub predictions: Option<String>,
    /// The selected metric names.
    pub metrics: Vec<String>,
    /// The prediction or matrix output format.
    pub output_format: String,
    /// Whether a flag, the environment or the workflow file chose
    /// [`Self::output_format`]; when false the file extension decides.
    #[serde(skip)]
    pub output_format_explicit: bool,
    /// The math backend kind recorded by the estimators.
    pub backend: String,
    /// Whether artifacts must not be written.
    pub dry_run: bool,
}

/// A sparse layer of configuration values, used for file, environment and flag
/// layers before they are merged onto the defaults.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SKConfigurationOverrides {
    /// Dataset path override.
    pub dataset: Option<String>,
    /// Dataset format override.
    pub dataset_format: Option<String>,
    /// Target column override.
    pub target: Option<String>,
    /// Estimator override.
    pub estimator: Option<String>,
    /// Ridge penalty override.
    pub alpha: Option<f64>,
    /// Model path override.
    pub model: Option<String>,
    /// Predictions path override.
    pub predictions: Option<String>,
    /// Metric list override.
    pub metrics: Option<Vec<String>>,
    /// Output format override.
    pub output_format: Option<String>,
    /// Backend override.
    pub backend: Option<String>,
    /// Dry-run override (only an explicit `--dry-run` sets it).
    pub dry_run: Option<bool>,
}

impl SKConfigurationOverrides {
    /// Fold `other` into this layer; values set by `other` win.
    pub fn merge(&mut self, other: SKConfigurationOverrides) {
        if other.dataset.is_some() {
            self.dataset = other.dataset;
        }
        if other.dataset_format.is_some() {
            self.dataset_format = other.dataset_format;
        }
        if other.target.is_some() {
            self.target = other.target;
        }
        if other.estimator.is_some() {
            self.estimator = other.estimator;
        }
        if other.alpha.is_some() {
            self.alpha = other.alpha;
        }
        if other.model.is_some() {
            self.model = other.model;
        }
        if other.predictions.is_some() {
            self.predictions = other.predictions;
        }
        if other.metrics.is_some() {
            self.metrics = other.metrics;
        }
        if other.output_format.is_some() {
            self.output_format = other.output_format;
        }
        if other.backend.is_some() {
            self.backend = other.backend;
        }
        if other.dry_run.is_some() {
            self.dry_run = other.dry_run;
        }
    }

    /// Merge this layer onto `configuration`, keeping every value it sets.
    pub fn apply(&self, mut configuration: SKEffectiveConfiguration) -> SKEffectiveConfiguration {
        if let Some(value) = &self.dataset {
            configuration.dataset = Some(value.clone());
        }
        if let Some(value) = &self.dataset_format {
            configuration.dataset_format = Some(value.clone());
        }
        if let Some(value) = &self.target {
            configuration.target = Some(value.clone());
        }
        if let Some(value) = &self.estimator {
            configuration.estimator = value.clone();
        }
        if let Some(value) = self.alpha {
            configuration.alpha = value;
        }
        if let Some(value) = &self.model {
            configuration.model = Some(value.clone());
        }
        if let Some(value) = &self.predictions {
            configuration.predictions = Some(value.clone());
        }
        if let Some(value) = &self.metrics {
            configuration.metrics = value.clone();
        }
        if let Some(value) = &self.output_format {
            configuration.output_format = value.clone();
        }
        if let Some(value) = &self.backend {
            configuration.backend = value.clone();
        }
        if let Some(value) = self.dry_run {
            configuration.dry_run = value;
        }
        configuration
    }
}
