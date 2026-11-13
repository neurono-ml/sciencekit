//! `sciencekit inspect` — describe a model artifact.

use std::path::Path;

use crate::cli::arguments::SKInspectArguments;
use crate::cli::commands::configuration::{
    sk_dry_run_requested, sk_report, sk_require_path, sk_resolve_command_configuration,
};
use crate::config::SKConfigurationOverrides;
use crate::error::SKCliError;
use crate::io::sk_inspect;

/// Execute `sciencekit inspect`.
pub fn sk_run_inspect(arguments: SKInspectArguments) -> Result<(), SKCliError> {
    let extra = SKConfigurationOverrides {
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
    let document = sk_inspect(Path::new(&model_path))?;
    sk_report(arguments.output.as_deref().map(Path::new), &document)
}
