//! OpenTelemetry trace, log and metric pipelines and `tracing` bridges
//! (spec `telemetry-export`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod logs;
mod traces;

pub use logs::{sk_build_log_layer, sk_build_logger_provider};
pub use traces::{sk_build_trace_layer, sk_build_tracer_provider};

#[cfg(test)]
mod logs_tests;
#[cfg(test)]
mod traces_tests;
