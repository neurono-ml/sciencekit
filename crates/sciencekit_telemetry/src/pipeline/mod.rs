//! OpenTelemetry trace, log and metric pipelines and `tracing` bridges
//! (spec `telemetry-export`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod layer;
mod logs;
mod metrics;
mod traces;

pub use layer::{SKTelemetryMetricsLayer, sk_build_metrics_layer};
pub use logs::{sk_build_log_layer, sk_build_logger_provider};
pub use metrics::{
    SK_METRIC_OPERATION_COLUMNS, SK_METRIC_OPERATION_COUNT, SK_METRIC_OPERATION_DURATION,
    SK_METRIC_OPERATION_ERRORS, SK_METRIC_OPERATION_ROWS, sk_build_meter_provider,
};
pub use traces::{sk_build_trace_layer, sk_build_tracer_provider};

#[cfg(test)]
mod logs_tests;
#[cfg(test)]
mod metrics_tests;
#[cfg(test)]
mod traces_tests;
