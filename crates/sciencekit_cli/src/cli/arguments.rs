//! The `clap` command surface (spec `workflows-cli`, requirement "Commands").

use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

/// The `sciencekit` command-line interface.
#[derive(Debug, Parser)]
#[command(
    name = "sciencekit",
    version,
    about = "Reproducible machine-learning workflows over the sciencekit library",
    long_about = "Fit, predict, transform, evaluate, export, inspect and run versioned \
                  workflow files. Configuration resolves as flags > environment > file > \
                  defaults, and every failure maps to a documented exit code."
)]
pub struct SKCliArguments {
    /// The command to execute.
    #[command(subcommand)]
    pub command: SKCliCommand,
}

/// Every documented subcommand.
#[derive(Debug, Subcommand)]
pub enum SKCliCommand {
    /// Fit an estimator or transformer and write a model artifact.
    Fit(SKFitArguments),
    /// Predict with a fitted model and write predictions.
    Predict(SKPredictArguments),
    /// Transform features with a fitted transformer.
    Transform(SKTransformArguments),
    /// Score a fitted model against a labelled dataset.
    Evaluate(SKEvaluateArguments),
    /// Convert a model artifact between formats.
    Export(SKExportArguments),
    /// Print the metadata of a model artifact.
    Inspect(SKInspectArguments),
    /// Execute a versioned workflow file.
    Run(SKRunArguments),
}

/// Configuration arguments shared by every command that resolves a config.
#[derive(Debug, Args, Default, Clone)]
pub struct SKConfigurationArguments {
    /// A workflow file that supplies the file-level configuration layer.
    #[arg(long)]
    pub config: Option<PathBuf>,
    /// Resolve and print the effective configuration without writing artifacts.
    #[arg(long)]
    pub dry_run: bool,
    /// Dataset format: csv or parquet.
    #[arg(long)]
    pub format: Option<String>,
    /// Target (label) column name.
    #[arg(long)]
    pub target: Option<String>,
    /// Estimator name: ridge, linear_regression or a scaler.
    #[arg(long)]
    pub estimator: Option<String>,
    /// Ridge penalty.
    #[arg(long)]
    pub alpha: Option<f64>,
    /// Math backend kind: faer (the only value without an accelerator feature).
    #[arg(long)]
    pub backend: Option<String>,
    /// Comma-separated metric names for evaluate and run.
    #[arg(long)]
    pub metrics: Option<String>,
    /// Prediction or matrix output format: csv, json or parquet.
    #[arg(long = "output-format")]
    pub output_format: Option<String>,
}

/// `sciencekit fit`.
#[derive(Debug, Args)]
pub struct SKFitArguments {
    /// Configuration layers and overrides.
    #[command(flatten)]
    pub configuration: SKConfigurationArguments,
    /// The input dataset path.
    #[arg(long)]
    pub input: Option<PathBuf>,
    /// The model artifact to write.
    #[arg(long)]
    pub model: Option<PathBuf>,
}

/// `sciencekit predict`.
#[derive(Debug, Args)]
pub struct SKPredictArguments {
    /// Configuration layers and overrides.
    #[command(flatten)]
    pub configuration: SKConfigurationArguments,
    /// The fitted model artifact.
    #[arg(long)]
    pub model: Option<PathBuf>,
    /// The input dataset path.
    #[arg(long)]
    pub input: Option<PathBuf>,
    /// The predictions file to write.
    #[arg(long)]
    pub output: Option<PathBuf>,
}

/// `sciencekit transform`.
#[derive(Debug, Args)]
pub struct SKTransformArguments {
    /// Configuration layers and overrides.
    #[command(flatten)]
    pub configuration: SKConfigurationArguments,
    /// The fitted transformer artifact.
    #[arg(long)]
    pub model: Option<PathBuf>,
    /// The input dataset path.
    #[arg(long)]
    pub input: Option<PathBuf>,
    /// The transformed matrix file to write.
    #[arg(long)]
    pub output: Option<PathBuf>,
}

/// `sciencekit evaluate`.
#[derive(Debug, Args)]
pub struct SKEvaluateArguments {
    /// Configuration layers and overrides.
    #[command(flatten)]
    pub configuration: SKConfigurationArguments,
    /// The fitted model artifact.
    #[arg(long)]
    pub model: Option<PathBuf>,
    /// The labelled dataset path.
    #[arg(long)]
    pub input: Option<PathBuf>,
    /// Where to write the metrics JSON; stdout when omitted.
    #[arg(long)]
    pub output: Option<PathBuf>,
}

/// `sciencekit export`.
#[derive(Debug, Args)]
pub struct SKExportArguments {
    /// Configuration layers and overrides.
    #[command(flatten)]
    pub configuration: SKConfigurationArguments,
    /// The source model artifact.
    #[arg(long)]
    pub model: Option<PathBuf>,
    /// The destination model artifact.
    #[arg(long)]
    pub output: Option<PathBuf>,
    /// Destination model format: safetensors, json or onnx.
    #[arg(long = "model-format")]
    pub model_format: Option<String>,
}

/// `sciencekit inspect`.
#[derive(Debug, Args)]
pub struct SKInspectArguments {
    /// Configuration layers and overrides.
    #[command(flatten)]
    pub configuration: SKConfigurationArguments,
    /// The model artifact to inspect.
    #[arg(long)]
    pub model: Option<PathBuf>,
    /// Where to write the inspection JSON; stdout when omitted.
    #[arg(long)]
    pub output: Option<PathBuf>,
}

/// `sciencekit run`.
#[derive(Debug, Args)]
pub struct SKRunArguments {
    /// Configuration layers and overrides.
    #[command(flatten)]
    pub configuration: SKConfigurationArguments,
    /// The versioned workflow file.
    pub workflow: PathBuf,
}
