//! OpenTelemetry log pipeline and `tracing` event bridge (spec `telemetry-export`).

use opentelemetry_appender_tracing::layer::OpenTelemetryTracingBridge;
use opentelemetry_sdk::logs::{BatchLogProcessor, LogExporter, SdkLoggerProvider};
use tracing::level_filters::LevelFilter;
use tracing_subscriber::Layer;
use tracing_subscriber::filter::Targets;
use tracing_subscriber::registry::LookupSpan;

use crate::configuration::SKTelemetryConfiguration;

/// Build an [`SdkLoggerProvider`] configured with batching and resource metadata.
pub fn sk_build_logger_provider<E>(
    configuration: &SKTelemetryConfiguration,
    exporter: E,
) -> SdkLoggerProvider
where
    E: LogExporter + 'static,
{
    let batch_processor = BatchLogProcessor::builder(exporter)
        .with_batch_config(configuration.batch().to_sdk_log_batch_configuration())
        .build();
    SdkLoggerProvider::builder()
        .with_log_processor(batch_processor)
        .with_resource(configuration.resource().to_sdk_resource())
        .build()
}

/// Build a `tracing` layer exporting structured events as OpenTelemetry logs
/// while filtering transport and internal SDK targets to prevent feedback loops.
pub fn sk_build_log_layer<S>(
    logger_provider: &SdkLoggerProvider,
) -> Box<dyn Layer<S> + Send + Sync + 'static>
where
    S: tracing::Subscriber + for<'span> LookupSpan<'span> + Send + Sync,
{
    let transport_filter = Targets::new()
        .with_default(LevelFilter::TRACE)
        .with_target("opentelemetry", LevelFilter::OFF)
        .with_target("hyper", LevelFilter::OFF)
        .with_target("tonic", LevelFilter::OFF)
        .with_target("h2", LevelFilter::OFF)
        .with_target("reqwest", LevelFilter::OFF);
    Box::new(OpenTelemetryTracingBridge::new(logger_provider).with_filter(transport_filter))
}
