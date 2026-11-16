//! The environment configuration layer (`SCIENCEKIT_*`).

use crate::config::effective::SKConfigurationOverrides;
use crate::error::SKCliError;

fn sk_read(name: &str) -> Option<String> {
    std::env::var(name).ok().filter(|value| !value.is_empty())
}

/// Read the environment layer.
///
/// # Errors
/// Returns a validation error when `SCIENCEKIT_ALPHA` is not a number.
pub fn sk_environment_overrides() -> Result<SKConfigurationOverrides, SKCliError> {
    let alpha = match sk_read("SCIENCEKIT_ALPHA") {
        Some(raw) => Some(raw.parse::<f64>().map_err(|_| {
            SKCliError::validation(
                "environment.SCIENCEKIT_ALPHA",
                format!("`{raw}` is not a number"),
            )
        })?),
        None => None,
    };
    Ok(SKConfigurationOverrides {
        dataset: sk_read("SCIENCEKIT_DATASET"),
        dataset_format: sk_read("SCIENCEKIT_DATASET_FORMAT"),
        target: sk_read("SCIENCEKIT_TARGET"),
        estimator: sk_read("SCIENCEKIT_ESTIMATOR"),
        alpha,
        model: sk_read("SCIENCEKIT_MODEL"),
        predictions: sk_read("SCIENCEKIT_PREDICTIONS"),
        metrics: sk_read("SCIENCEKIT_METRICS").map(|raw| sk_split_metrics(&raw)),
        output_format: sk_read("SCIENCEKIT_OUTPUT_FORMAT"),
        backend: sk_read("SCIENCEKIT_BACKEND"),
        dry_run: None,
    })
}

/// Split and trim a comma-separated metric list.
pub fn sk_split_metrics(raw: &str) -> Vec<String> {
    raw.split(',')
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_string)
        .collect()
}
