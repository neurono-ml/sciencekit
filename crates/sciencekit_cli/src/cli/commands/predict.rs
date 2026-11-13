//! `sciencekit predict` — predict with a fitted model.

use std::path::Path;

use serde_json::json;

use crate::cli::arguments::SKPredictArguments;
use crate::cli::commands::configuration::{
    sk_dry_run_requested, sk_report, sk_require_path, sk_resolve_command_configuration,
};
use crate::cli::commands::support::sk_output_format;
use crate::config::SKConfigurationOverrides;
use crate::error::SKCliError;
use crate::io::{sk_load_dataset, sk_load_native, sk_write_predictions};

/// Execute `sciencekit predict`.
pub fn sk_run_predict(arguments: SKPredictArguments) -> Result<(), SKCliError> {
    let extra = SKConfigurationOverrides {
        dataset: arguments
            .input
            .as_ref()
            .map(|path| path.display().to_string()),
        model: arguments
            .model
            .as_ref()
            .map(|path| path.display().to_string()),
        predictions: arguments
            .output
            .as_ref()
            .map(|path| path.display().to_string()),
        ..SKConfigurationOverrides::default()
    };
    let configuration = sk_resolve_command_configuration(&arguments.configuration, extra)?;
    if sk_dry_run_requested(&configuration)? {
        return Ok(());
    }

    let model_path = sk_require_path(
        configuration.model.as_deref(),
        "--model",
        "configuration.model",
    )?;
    let dataset_path = sk_require_path(
        configuration.dataset.as_deref(),
        "--input",
        "configuration.dataset",
    )?;
    let output_path = sk_require_path(
        configuration.predictions.as_deref(),
        "--output",
        "configuration.predictions",
    )?;
    let format = sk_output_format(&configuration, Path::new(&output_path))?;

    let dataset = sk_load_dataset(
        Path::new(&dataset_path),
        configuration.dataset_format.as_deref(),
        configuration.target.as_deref(),
    )?;
    let model = sk_load_native(Path::new(&model_path))?;
    let predictions = model
        .predict_regression(dataset.features.view())
        .map_err(|error| SKCliError::runtime(format!("prediction: {error}")))?;
    let target_name = configuration
        .target
        .clone()
        .unwrap_or_else(|| "prediction".to_string());
    let values: Vec<f64> = predictions.iter().copied().collect();
    sk_write_predictions(Path::new(&output_path), format, &target_name, &values)?;

    sk_report(
        None,
        &json!({
            "command": "predict",
            "model": model_path,
            "number_of_rows": dataset.number_of_rows(),
            "output": output_path,
            "format": format_name(format),
        }),
    )
}

fn format_name(format: crate::io::SKOutputFormat) -> &'static str {
    match format {
        crate::io::SKOutputFormat::Csv => "csv",
        crate::io::SKOutputFormat::Json => "json",
        crate::io::SKOutputFormat::Parquet => "parquet",
    }
}
