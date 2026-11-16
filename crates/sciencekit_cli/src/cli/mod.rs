//! The command-line layer: arguments, dispatch and the exit-code contract.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares submodules and
//! re-exports their public items; every implementation lives in its own file.

pub mod arguments;
pub mod commands;
pub mod exit_codes;

#[allow(unused_imports)]
pub use arguments::SKCliArguments;
#[allow(unused_imports)]
pub use exit_codes::SKExitCode;

#[cfg(test)]
mod exit_codes_tests;
