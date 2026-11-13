//! `sciencekit transform` — apply a fitted transformer.

use std::path::Path;

use serde_json::json;

use crate::cli::arguments::SKTransformArguments;
use crate::cli::commands::configuration::{
    sk_dry_run_requested, sk_report, sk_require_path, sk_resolve_command_configuration,
};
use crate::cli::commands::support::sk_output_format;
use crate::config::SKConfigurationOverrides;
use crate::error::SKCliError;
use crate::io::{sk_load_dataset, sk_load_native, sk_write_transformed_matrix};

/// Execute `sciencekit transform`.
pub fn sk_run_transform(arguments: SKTransformArguments) -> Result<(), SKCliError> {
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
    let transformed = model
        .transform(dataset.features.view())
        .map_err(|error| SKCliError::runtime(format!("transform: {error}")))?;
    let matrix: Vec<Vec<f64>> = transformed
        .rows()
        .into_iter()
        .map(|row| row.iter().copied().collect())
        .collect();
    let names: Vec<String> = (0..transformed.ncols())
        .map(|position| format!("feature_{position}"))
        .collect();
    sk_write_transformed_matrix(Path::new(&output_path), format, &names, &matrix)?;

    sk_report(
        None,
        &json!({
            "command": "transform",
            "model": model_path,
            "number_of_rows": transformed.nrows(),
            "number_of_features": transformed.ncols(),
            "output": output_path,
        }),
    )
}
