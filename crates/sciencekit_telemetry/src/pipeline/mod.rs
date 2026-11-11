//! OpenTelemetry trace, log and metric pipelines and `tracing` bridges
//! (spec `telemetry-export`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod traces;

pub use traces::{sk_build_trace_layer, sk_build_tracer_provider};

#[cfg(test)]
mod traces_tests;
