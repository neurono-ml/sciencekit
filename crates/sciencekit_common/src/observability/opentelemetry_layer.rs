//! Opt-in OpenTelemetry layer (spec `observability`, PRD §9.2).

/// Build an OpenTelemetry tracing layer over the configured global tracer.
///
/// Available only with the `observability-export` feature. The consumer
/// configures the OpenTelemetry tracer provider (e.g. an OTLP exporter) and
/// combines the returned layer with their `tracing_subscriber` registry; the
/// default build compiles this out and never initializes an exporter.
pub fn sk_opentelemetry_layer<S>(
    tracer_name: &str,
) -> Box<dyn tracing_subscriber::Layer<S> + Send + Sync + 'static>
where
    S: tracing::Subscriber
        + for<'span> tracing_subscriber::registry::LookupSpan<'span>
        + Send
        + Sync,
{
    Box::new(
        tracing_opentelemetry::layer::<S>()
            .with_tracer(opentelemetry::global::tracer(tracer_name.to_owned())),
    )
}
