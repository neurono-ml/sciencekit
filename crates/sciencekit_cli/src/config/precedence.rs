//! Configuration precedence: flags > environment > file > defaults.

use crate::cli::arguments::SKConfigurationArguments;
use crate::cli::commands::estimator::sk_is_transformer;
use crate::config::backend::sk_parse_backend;
use crate::config::defaults::{
    SK_DEFAULT_ALPHA, SK_DEFAULT_BACKEND, SK_DEFAULT_ESTIMATOR, SK_DEFAULT_OUTPUT_FORMAT,
    sk_default_metrics,
};
use crate::config::effective::{SKConfigurationOverrides, SKEffectiveConfiguration};
use crate::config::environment::sk_split_metrics;
use crate::error::SKCliError;
use crate::io::{SKDatasetFormat, SKOutputFormat};
use crate::workflow::schema::SKWorkflow;
use crate::workflow::validation::SK_SUPPORTED_METRICS;

/// The documented default layer.
pub fn sk_default_configuration() -> SKEffectiveConfiguration {
    SKEffectiveConfiguration {
        dataset: None,
        dataset_format: None,
        target: None,
        estimator: SK_DEFAULT_ESTIMATOR.to_string(),
        alpha: SK_DEFAULT_ALPHA,
        model: None,
        predictions: None,
        metrics: sk_default_metrics(),
        output_format: SK_DEFAULT_OUTPUT_FORMAT.to_string(),
        output_format_explicit: false,
        backend: SK_DEFAULT_BACKEND.to_string(),
        dry_run: false,
    }
}

/// The file layer carried by a parsed workflow document.
pub fn sk_file_overrides(workflow: &SKWorkflow) -> SKConfigurationOverrides {
    SKConfigurationOverrides {
        dataset: workflow
            .dataset
            .as_ref()
            .map(|dataset| dataset.path.clone()),
        dataset_format: workflow
            .dataset
            .as_ref()
            .and_then(|dataset| dataset.format.clone()),
        target: workflow
            .dataset
            .as_ref()
            .and_then(|dataset| dataset.target.clone()),
        estimator: workflow
            .estimator
            .as_ref()
            .map(|estimator| estimator.name.clone()),
        alpha: workflow
            .estimator
            .as_ref()
            .and_then(|estimator| estimator.alpha),
        model: workflow
            .output
            .as_ref()
            .and_then(|output| output.model.clone()),
        predictions: workflow
            .output
            .as_ref()
            .and_then(|output| output.predictions.clone()),
        metrics: (!workflow.metrics.is_empty()).then(|| workflow.metrics.clone()),
        output_format: workflow
            .output
            .as_ref()
            .and_then(|output| output.format.clone()),
        backend: workflow
            .estimator
            .as_ref()
            .and_then(|estimator| estimator.backend.clone()),
        dry_run: None,
    }
}

/// The flag layer carried by the shared configuration arguments.
pub fn sk_flag_overrides(arguments: &SKConfigurationArguments) -> SKConfigurationOverrides {
    SKConfigurationOverrides {
        dataset: None,
        dataset_format: arguments.format.clone(),
        target: arguments.target.clone(),
        estimator: arguments.estimator.clone(),
        alpha: arguments.alpha,
        model: None,
        predictions: None,
        metrics: arguments
            .metrics
            .as_ref()
            .map(|raw| sk_split_metrics(raw))
            .filter(|metrics| !metrics.is_empty()),
        output_format: arguments.output_format.clone(),
        backend: arguments.backend.clone(),
        dry_run: arguments.dry_run.then_some(true),
    }
}

/// Resolve defaults, the optional file layer, the environment and flags.
pub fn sk_resolve_configuration(
    workflow: Option<&SKWorkflow>,
    environment: &SKConfigurationOverrides,
    flags: &SKConfigurationOverrides,
) -> SKEffectiveConfiguration {
    let with_file = match workflow {
        Some(workflow) => sk_file_overrides(workflow).apply(sk_default_configuration()),
        None => sk_default_configuration(),
    };
    let file_sets_output_format = workflow
        .and_then(|workflow| workflow.output.as_ref())
        .and_then(|output| output.format.as_ref())
        .is_some();
    let mut configuration = flags.apply(environment.apply(with_file));
    configuration.output_format_explicit = file_sets_output_format
        || environment.output_format.is_some()
        || flags.output_format.is_some();
    configuration
}

/// Validate the config values that do not depend on reading an artifact.
///
/// This runs for every command before execution and before `--dry-run`, so a
/// configuration that resolves cleanly is always executable: format names,
/// estimator names, metric names and the backend are checked here instead of
/// being discovered halfway through a command.
pub fn sk_validate_configuration(
    configuration: &SKEffectiveConfiguration,
) -> Result<(), SKCliError> {
    if configuration.alpha <= 0.0 || !configuration.alpha.is_finite() {
        return Err(SKCliError::validation(
            "configuration.alpha",
            "the penalty must be a finite positive number",
        ));
    }
    if configuration.estimator.trim().is_empty() {
        return Err(SKCliError::validation(
            "configuration.estimator",
            "an estimator name is required",
        ));
    }
    if !sk_is_transformer(&configuration.estimator)
        && !matches!(
            configuration.estimator.as_str(),
            "ridge" | "linear_regression"
        )
    {
        return Err(SKCliError::validation(
            "configuration.estimator",
            format!(
                "`{}` is not a supported estimator; use ridge, linear_regression, \
                 standard_scaler, minmax_scaler or robust_scaler",
                configuration.estimator
            ),
        ));
    }
    if configuration.metrics.is_empty() {
        return Err(SKCliError::validation(
            "configuration.metrics",
            "at least one metric is required",
        ));
    }
    for metric in &configuration.metrics {
        if !SK_SUPPORTED_METRICS.contains(&metric.as_str()) {
            return Err(SKCliError::validation(
                "configuration.metrics",
                format!("`{metric}` is not a supported metric; use r2, rmse, mse or mae"),
            ));
        }
    }
    if let Some(format) = &configuration.dataset_format {
        SKDatasetFormat::from_name(format)?;
    }
    SKOutputFormat::from_name(&configuration.output_format)?;
    sk_parse_backend(&configuration.backend)?;
    Ok(())
}
