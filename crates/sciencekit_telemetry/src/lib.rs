//! `sciencekit_telemetry` — opt-in OpenTelemetry trace, log and metric export
//! pipelines and lifecycle for `sciencekit` (spec `telemetry-export`, PRD §9.2).
//!
//! Library crates emit lightweight [`tracing`] spans and events through
//! [`sciencekit_common::observability`] without depending on an OpenTelemetry
//! SDK or network transport. Downstream applications that want production OTLP
//! telemetry depend on `sciencekit_telemetry`, build a validated
//! [`SKTelemetryConfiguration`] and initialize the global pipeline once at
//! startup.
//!
//! **Naming:** public items follow PRD §3.4 — structs, enums and traits are
//! prefixed `SK`, free-scope functions and constants use `sk_`/`SK_`, methods
//! carry no prefix, and no Rust identifier uses abbreviations. Folder `mod.rs`
//! files are pure dispatchers; companion `*_tests.rs` modules hold the TDD
//! suites.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod configuration;
pub mod pipeline;

pub use configuration::{
    SKTelemetryBatchConfiguration, SKTelemetryBatchConfigurationBuilder, SKTelemetryConfiguration,
    SKTelemetryConfigurationBuilder, SKTelemetryResource, SKTelemetrySampler, SKTelemetryTransport,
};
pub use pipeline::{sk_build_trace_layer, sk_build_tracer_provider};
