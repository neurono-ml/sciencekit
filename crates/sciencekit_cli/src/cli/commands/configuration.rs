//! Shared command plumbing: configuration resolution, dry-run and reporting.

use std::path::Path;

use serde_json::Value;

use crate::cli::arguments::SKConfigurationArguments;
use crate::config::effective::{SKConfigurationOverrides, SKEffectiveConfiguration};
use crate::config::environment::sk_environment_overrides;
use crate::config::precedence::{
    sk_flag_overrides, sk_resolve_configuration, sk_validate_configuration,
};
use crate::config::{sk_render_dry_run, sk_validate_dry_run_paths};
use crate::error::SKCliError;
use crate::workflow::loader::sk_load_workflow;

/// Resolve the configuration for one command, with command-specific path flags.
pub fn sk_resolve_command_configuration(
    arguments: &SKConfigurationArguments,
    extra_flags: SKConfigurationOverrides,
) -> Result<SKEffectiveConfiguration, SKCliError> {
    let workflow = match &arguments.config {
        Some(path) => Some(sk_load_workflow(path)?),
        None => None,
    };
    let environment = sk_environment_overrides()?;
    let mut flags = sk_flag_overrides(arguments);
    flags.merge(extra_flags);
    let configuration = sk_resolve_configuration(workflow.as_ref(), &environment, &flags);
    sk_validate_configuration(&configuration)?;
    Ok(configuration)
}

/// Print the resolved configuration and stop when `--dry-run` is set.
///
/// Returns `true` when the command must not execute.
pub fn sk_dry_run_requested(configuration: &SKEffectiveConfiguration) -> Result<bool, SKCliError> {
    if !configuration.dry_run {
        return Ok(false);
    }
    sk_validate_dry_run_paths(configuration)?;
    println!("{}", sk_render_dry_run(configuration)?);
    Ok(true)
}

/// Require a configured path, naming the flag that supplies it.
pub fn sk_require_path(value: Option<&str>, flag: &str, path: &str) -> Result<String, SKCliError> {
    value.map(str::to_string).ok_or_else(|| {
        SKCliError::usage(format!(
            "`{flag}` is required; supply the flag, an environment variable or a workflow file ({path})"
        ))
    })
}

/// Print a JSON report to stdout, or write it to `path`.
pub fn sk_report(path: Option<&Path>, value: &Value) -> Result<(), SKCliError> {
    let text = serde_json::to_string_pretty(value)
        .map_err(|error| SKCliError::runtime(format!("report serialization: {error}")))?;
    match path {
        Some(path) => std::fs::write(path, text)
            .map_err(|error| SKCliError::data(path.display().to_string(), error.to_string())),
        None => {
            println!("{text}");
            Ok(())
        }
    }
}
