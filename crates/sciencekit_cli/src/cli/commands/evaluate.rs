//! `sciencekit evaluate` — score a fitted model on a labelled dataset.

use std::path::Path;

use serde_json::json;

use crate::cli::arguments::SKEvaluateArguments;
use crate::cli::commands::configuration::{
    sk_dry_run_requested, sk_report, sk_require_path, sk_resolve_command_configuration,
};
use crate::cli::commands::estimator::sk_require_targets;
use crate::cli::commands::metrics::sk_compute_metrics;
use crate::config::SKConfigurationOverrides;
use crate::error::SKCliError;
use crate::io::sk_load_dataset;
use crate::io::sk_load_native;

/// The output formats the evaluate command accepts for its metrics document.
fn sk_resolve_metrics_format(
    arguments: &SKEvaluateArguments,
    output: Option<&str>,
) -> Result<(), SKCliError> {
    if let Some(output) = output {
        if !output.ends_with(".json") {
            return Err(SKCliError::validation(
                "configuration.metrics_output",
                "the metrics document must use the `.json` extension",
            ));
        }
    }
    let _ = arguments;
    Ok(())
}

/// Execute `sciencekit evaluate`.
pub fn sk_run_evaluate(arguments: SKEvaluateArguments) -> Result<(), SKCliError> {
    let extra = SKConfigurationOverrides {
        dataset: arguments
            .input
            .as_ref()
            .map(|path| path.display().to_string()),
        model: arguments
            .model
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
    let target_name = sk_require_path(
        configuration.target.as_deref(),
        "--target",
        "configuration.target",
    )?;
    let output_path = arguments
        .output
        .as_ref()
        .map(|path| path.display().to_string());
    sk_resolve_metrics_format(&arguments, output_path.as_deref())?;

    let dataset = sk_load_dataset(
        Path::new(&dataset_path),
        configuration.dataset_format.as_deref(),
        Some(&target_name),
    )?;
    let targets = sk_require_targets(dataset.targets.as_ref())?;
    let model = sk_load_native(Path::new(&model_path))?;
    let predictions = model
        .predict_regression(dataset.features.view())
        .map_err(|error| SKCliError::runtime(format!("prediction: {error}")))?;
    let metrics = sk_compute_metrics(targets, predictions.view(), &configuration.metrics)?;

    sk_report(
        output_path.as_deref().map(Path::new),
        &json!({
            "command": "evaluate",
            "model": model_path,
            "number_of_rows": dataset.number_of_rows(),
            "metrics": metrics,
        }),
    )
}
