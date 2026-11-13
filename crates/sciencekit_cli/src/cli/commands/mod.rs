//! Command implementations.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares submodules and
//! re-exports their public entry points; every command lives in its own file.

pub(crate) mod configuration;
pub(crate) mod estimator;
mod evaluate;
mod export;
mod fit;
mod inspect;
pub(crate) mod metrics;
mod predict;
mod run;
pub(crate) mod support;
mod transform;

pub use evaluate::sk_run_evaluate;
pub use export::sk_run_export;
pub use fit::sk_run_fit;
pub use inspect::sk_run_inspect;
pub use predict::sk_run_predict;
pub use run::sk_run_workflow_command;
pub use transform::sk_run_transform;

use crate::cli::arguments::{SKCliArguments, SKCliCommand};
use crate::error::SKCliError;

/// Dispatch one parsed command.
pub fn sk_run(arguments: SKCliArguments) -> Result<(), SKCliError> {
    match arguments.command {
        SKCliCommand::Fit(arguments) => sk_run_fit(arguments),
        SKCliCommand::Predict(arguments) => sk_run_predict(arguments),
        SKCliCommand::Transform(arguments) => sk_run_transform(arguments),
        SKCliCommand::Evaluate(arguments) => sk_run_evaluate(arguments),
        SKCliCommand::Export(arguments) => sk_run_export(arguments),
        SKCliCommand::Inspect(arguments) => sk_run_inspect(arguments),
        SKCliCommand::Run(arguments) => sk_run_workflow_command(arguments),
    }
}
