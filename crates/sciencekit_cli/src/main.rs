//! `sciencekit` — the command-line entry point over the sciencekit library.
//!
//! The binary is a thin orchestration layer: it parses arguments, resolves the
//! effective configuration, initializes optional telemetry, dispatches one
//! command and maps every error to a stable exit code. No estimator, metric or
//! serialization format is reimplemented here.
//!
//! **Naming:** public items follow PRD §3.4 — structs and enums are prefixed
//! `SK`, free-scope functions use `sk_`, methods carry no prefix. Folder
//! `mod.rs` files are pure dispatchers; companion `*_tests.rs` modules hold the
//! integration suites and run the compiled binary.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

mod cli;
mod config;
mod error;
mod io;
mod telemetry;
#[cfg(test)]
mod telemetry_tests;
mod workflow;

use std::process::ExitCode;

use clap::Parser;

use cli::arguments::SKCliArguments;
use cli::commands::sk_run;

fn main() -> ExitCode {
    let arguments = SKCliArguments::parse();
    let telemetry_guard = match telemetry::sk_initialize_telemetry() {
        Ok(guard) => guard,
        Err(error) => {
            eprintln!("error: {error}");
            return ExitCode::from(error.exit_code().code());
        }
    };

    match sk_run(arguments) {
        Ok(()) => {
            drop(telemetry_guard);
            ExitCode::from(cli::SKExitCode::Success.code())
        }
        Err(error) => {
            eprintln!("error: {error}");
            drop(telemetry_guard);
            ExitCode::from(error.exit_code().code())
        }
    }
}
