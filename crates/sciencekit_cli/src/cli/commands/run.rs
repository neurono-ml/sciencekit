//! The `run` command: load, resolve and execute a versioned workflow.

use crate::cli::arguments::SKRunArguments;
use crate::cli::commands::configuration::{sk_dry_run_requested, sk_report};
use crate::config::{
    sk_environment_overrides, sk_flag_overrides, sk_resolve_configuration,
    sk_validate_configuration,
};
use crate::error::SKCliError;
use crate::workflow::loader::sk_load_workflow;
use crate::workflow::runner::sk_execute_workflow;

/// Execute `sciencekit run`.
pub fn sk_run_workflow_command(arguments: SKRunArguments) -> Result<(), SKCliError> {
    let workflow = sk_load_workflow(&arguments.workflow)?;
    let environment = sk_environment_overrides()?;
    let flags = sk_flag_overrides(&arguments.configuration);
    let configuration = sk_resolve_configuration(Some(&workflow), &environment, &flags);
    sk_validate_configuration(&configuration)?;
    if sk_dry_run_requested(&configuration)? {
        return Ok(());
    }
    let report = sk_execute_workflow(&workflow, &configuration)?;
    sk_report(None, &report)
}
