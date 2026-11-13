//! `sciencekit export` — convert a model artifact between formats.

use std::path::Path;

use serde_json::json;

use crate::cli::arguments::SKExportArguments;
use crate::cli::commands::configuration::{
    sk_dry_run_requested, sk_report, sk_require_path, sk_resolve_command_configuration,
};
use crate::config::SKConfigurationOverrides;
use crate::error::SKCliError;
use crate::io::{SKModelFormat, sk_load_state, sk_save_state};

/// Execute `sciencekit export`.
pub fn sk_run_export(arguments: SKExportArguments) -> Result<(), SKCliError> {
    let extra = SKConfigurationOverrides {
        model: arguments
            .model
            .as_ref()
            .map(|path| path.display().to_string()),
        ..SKConfigurationOverrides::default()
    };
    let configuration = sk_resolve_command_configuration(&arguments.configuration, extra)?;
    let flag_format = match &arguments.model_format {
        Some(name) => Some(SKModelFormat::from_name(name)?),
        None => None,
    };
    if sk_dry_run_requested(&configuration)? {
        return Ok(());
    }

    let model_path = sk_require_path(
        configuration.model.as_deref(),
        "--model",
        "configuration.model",
    )?;
    let output_path = sk_require_path(
        arguments
            .output
            .as_ref()
            .map(|path| path.display().to_string())
            .as_deref(),
        "--output",
        "command.output",
    )?;
    let format = match flag_format {
        Some(format) => format,
        None => SKModelFormat::from_path(Path::new(&output_path)),
    };

    let state = sk_load_state(Path::new(&model_path))?;
    sk_save_state(Path::new(&output_path), &state, format)?;

    sk_report(
        None,
        &json!({
            "command": "export",
            "model": model_path,
            "output": output_path,
            "format": format.name(),
            "estimator": state.metadata().estimator_name(),
            "tensor_count": state.number_of_tensors(),
        }),
    )
}
