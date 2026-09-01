//! Observability for library operations (spec `observability`, PRD §9.2).
//!
//! Public operations emit `tracing` spans with structured fields (operation,
//! shape, execution mode, backend) and record the outcome (duration, error) on
//! the span. [`sk_run_operation`] is the reusable wrapper algorithm crates
//! call around fit, transform and predict. OpenTelemetry export is opt-in
//! behind the `observability-export` feature; the default build initializes no
//! exporter and emits no OTLP traffic.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod operation_attributes;
mod operation_observation;
mod run_operation;

#[cfg(feature = "observability-export")]
mod opentelemetry_layer;

#[cfg(test)]
mod observability_tests;

pub use operation_attributes::SKOperationAttributes;
pub use operation_observation::SKOperationObservation;
pub use run_operation::sk_run_operation;

#[cfg(feature = "observability-export")]
pub use opentelemetry_layer::sk_opentelemetry_layer;
