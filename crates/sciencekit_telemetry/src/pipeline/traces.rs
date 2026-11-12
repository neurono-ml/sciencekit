//! OpenTelemetry trace pipeline and `tracing` span bridge (spec `telemetry-export`).

use std::fmt;
use std::future::Future;
use std::time::Duration;

use opentelemetry::Value;
use opentelemetry::trace::{Status, TracerProvider as _};
use opentelemetry_sdk::Resource;
use opentelemetry_sdk::error::OTelSdkResult;
use opentelemetry_sdk::trace::{BatchSpanProcessor, SdkTracerProvider, SpanData, SpanExporter};
use tracing::level_filters::LevelFilter;
use tracing_subscriber::Layer;
use tracing_subscriber::filter::Targets;
use tracing_subscriber::registry::LookupSpan;

use crate::configuration::SKTelemetryConfiguration;

/// Wraps a [`SpanExporter`] so operation spans carrying a recorded `error`
/// field also surface an OpenTelemetry [`Status::Error`] status with the
/// failure message.
pub(crate) struct SKOperationSpanExporter<E> {
    inner: E,
}

impl<E> SKOperationSpanExporter<E> {
    pub(crate) fn new(inner: E) -> Self {
        Self { inner }
    }
}

impl<E: fmt::Debug> fmt::Debug for SKOperationSpanExporter<E> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SKOperationSpanExporter")
            .field("inner", &self.inner)
            .finish()
    }
}

fn enrich_span_error_status(span: &mut SpanData) {
    let needs_error_status = match &span.status {
        Status::Unset => true,
        Status::Error { description } => description.is_empty(),
        Status::Ok => false,
    };
    if !needs_error_status {
        return;
    }
    let recorded_error = span
        .attributes
        .iter()
        .find(|entry| entry.key.as_str() == "error")
        .and_then(|entry| match &entry.value {
            Value::String(text) if !text.as_str().is_empty() => Some(text.to_string()),
            _ => None,
        });
    if let Some(error_message) = recorded_error {
        span.status = Status::error(error_message);
    }
}

impl<E> SpanExporter for SKOperationSpanExporter<E>
where
    E: SpanExporter,
{
    fn export(&self, mut batch: Vec<SpanData>) -> impl Future<Output = OTelSdkResult> + Send {
        batch.iter_mut().for_each(enrich_span_error_status);
        self.inner.export(batch)
    }

    fn shutdown_with_timeout(&self, timeout: Duration) -> OTelSdkResult {
        self.inner.shutdown_with_timeout(timeout)
    }

    fn force_flush(&self) -> OTelSdkResult {
        self.inner.force_flush()
    }

    fn set_resource(&mut self, resource: &Resource) {
        self.inner.set_resource(resource);
    }
}

/// Build an [`SdkTracerProvider`] configured with batching, sampling and resource metadata.
pub fn sk_build_tracer_provider<E>(
    configuration: &SKTelemetryConfiguration,
    exporter: E,
) -> SdkTracerProvider
where
    E: SpanExporter + 'static,
{
    let wrapped_exporter = SKOperationSpanExporter::new(exporter);
    let batch_processor = BatchSpanProcessor::builder(wrapped_exporter)
        .with_batch_config(configuration.batch().to_sdk_trace_batch_configuration())
        .build();
    SdkTracerProvider::builder()
        .with_span_processor(batch_processor)
        .with_sampler(configuration.sampler().to_sdk_sampler())
        .with_resource(configuration.resource().to_sdk_resource())
        .build()
}

/// Build a `tracing` layer exporting spans to `tracer_provider` while silencing
/// internal HTTP/gRPC transport targets to prevent export feedback loops.
pub fn sk_build_trace_layer<S>(
    tracer_provider: &SdkTracerProvider,
) -> Box<dyn Layer<S> + Send + Sync + 'static>
where
    S: tracing::Subscriber + for<'span> LookupSpan<'span> + Send + Sync,
{
    let tracer = tracer_provider.tracer("sciencekit");
    let transport_filter = Targets::new()
        .with_default(LevelFilter::TRACE)
        .with_target("hyper", LevelFilter::OFF)
        .with_target("tonic", LevelFilter::OFF)
        .with_target("h2", LevelFilter::OFF)
        .with_target("reqwest", LevelFilter::OFF);
    Box::new(
        tracing_opentelemetry::layer::<S>()
            .with_tracer(tracer)
            .with_filter(transport_filter),
    )
}
