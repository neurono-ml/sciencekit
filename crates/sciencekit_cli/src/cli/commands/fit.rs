//! `sciencekit fit` — fit one estimator or transformer.

use std::path::Path;

use serde_json::json;

use crate::cli::arguments::SKFitArguments;
use crate::cli::commands::configuration::{
    sk_dry_run_requested, sk_report, sk_require_path, sk_resolve_command_configuration,
};
use crate::cli::commands::estimator::{
    sk_fit_regressor, sk_fit_transformer, sk_is_transformer, sk_require_targets,
};
use crate::config::SKConfigurationOverrides;
use crate::config::sk_parse_backend;
use crate::error::SKCliError;
use crate::io::{SKModelFormat, sk_load_dataset, sk_save_state};

/// Execute `sciencekit fit`.
pub fn sk_run_fit(arguments: SKFitArguments) -> Result<(), SKCliError> {
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

    let dataset_path = sk_require_path(
        configuration.dataset.as_deref(),
        "--input",
        "configuration.dataset",
    )?;
    let model_path = sk_require_path(
        configuration.model.as_deref(),
        "--model",
        "configuration.model",
    )?;
    let backend = sk_parse_backend(&configuration.backend)?;
    let transformer = sk_is_transformer(&configuration.estimator);
    let dataset = sk_load_dataset(
        Path::new(&dataset_path),
        configuration.dataset_format.as_deref(),
        configuration.target.as_deref(),
    )?;

    let state = if transformer {
        sk_fit_transformer(&configuration.estimator, backend, dataset.features.view())?
    } else {
        let targets = sk_require_targets(dataset.targets.as_ref())?;
        sk_fit_regressor(
            &configuration.estimator,
            configuration.alpha,
            backend,
            dataset.features.view(),
            targets,
        )?
    };
    sk_save_state(
        Path::new(&model_path),
        &state,
        SKModelFormat::from_path(Path::new(&model_path)),
    )?;

    sk_report(
        None,
        &json!({
            "command": "fit",
            "estimator": configuration.estimator,
            "alpha": configuration.alpha,
            "backend": configuration.backend,
            "number_of_rows": dataset.number_of_rows(),
            "number_of_features": dataset.number_of_features(),
            "feature_names": dataset.feature_names,
            "target": dataset.target_name,
            "model": model_path,
        }),
    )
}
