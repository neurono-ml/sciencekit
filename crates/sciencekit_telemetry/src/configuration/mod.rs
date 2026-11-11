//! Configuration, resource attributes, transport and sampling types for
//! OpenTelemetry export (spec `telemetry-export`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod resource;
mod sampler;
mod transport;

pub use resource::SKTelemetryResource;
pub use sampler::SKTelemetrySampler;
pub use transport::SKTelemetryTransport;

#[cfg(test)]
mod configuration_tests;
