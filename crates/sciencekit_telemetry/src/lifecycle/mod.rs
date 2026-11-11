//! Global telemetry lifecycle: initialization, flush, shutdown and drop safety
//! (spec `telemetry-export`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod global_state;
mod init;
mod shutdown;

pub use init::SKTelemetry;

#[cfg(test)]
mod concurrency_tests;
#[cfg(test)]
mod flush_timeout_tests;
#[cfg(test)]
mod lifecycle_tests;
#[cfg(test)]
mod noop_tests;
