//! Configuration, resource attributes, transport and sampling types for
//! OpenTelemetry export (spec `telemetry-export`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod batch;
mod batch_builder;
mod builder;
mod resource;
mod sampler;
mod telemetry_configuration;
mod transport;
mod validation;

pub use batch::SKTelemetryBatchConfiguration;
pub use batch_builder::SKTelemetryBatchConfigurationBuilder;
pub use builder::SKTelemetryConfigurationBuilder;
pub use resource::SKTelemetryResource;
pub use sampler::SKTelemetrySampler;
pub use telemetry_configuration::SKTelemetryConfiguration;
pub use transport::SKTelemetryTransport;

#[cfg(test)]
mod configuration_tests;
#[cfg(test)]
mod validation_tests;
