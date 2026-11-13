//! The `telemetry` family: opt-in OpenTelemetry export.
//!
//! Every item here resolves to `sciencekit_telemetry`. The umbrella forwards
//! the `telemetry` feature to this module so a core-only build pulls in no
//! OpenTelemetry SDK.

pub use sciencekit_telemetry::SKTelemetry;
pub use sciencekit_telemetry::{
    SK_METRIC_OPERATION_COLUMNS, SK_METRIC_OPERATION_COUNT, SK_METRIC_OPERATION_DURATION,
    SK_METRIC_OPERATION_ERRORS, SK_METRIC_OPERATION_ROWS, SKTelemetryMetricsLayer,
    SKTelemetryPipeline, sk_build_log_layer, sk_build_logger_provider, sk_build_meter_provider,
    sk_build_metrics_layer, sk_build_otlp_pipeline, sk_build_trace_layer, sk_build_tracer_provider,
};
pub use sciencekit_telemetry::{
    SKTelemetryBatchConfiguration, SKTelemetryBatchConfigurationBuilder, SKTelemetryConfiguration,
    SKTelemetryConfigurationBuilder, SKTelemetryResource, SKTelemetrySampler, SKTelemetryTransport,
};
pub use sciencekit_telemetry::{configuration, lifecycle, pipeline};
