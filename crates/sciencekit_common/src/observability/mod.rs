//! Observability for library operations (spec `observability`, PRD §9.2).
//!
//! Public operations emit `tracing` spans with structured fields (operation,
//! shape, execution mode, backend) and record the outcome (duration, error) on
//! the span. [`sk_run_operation`] is the reusable wrapper algorithm crates
//! call around fit, transform and predict. OpenTelemetry export is opt-in
//! behind the `observability-export` feature; the default build initializes no
//! exporter and emits no OTLP traffic.

use std::time::Instant;

use crate::execution::SKExecutionMode;

/// Structured attributes recorded on an operation span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SKOperationAttributes {
    /// The operation name (`fit`, `transform`, `predict`, ...).
    pub operation: &'static str,
    /// The number of rows of the operation input.
    pub rows: usize,
    /// The number of columns of the operation input.
    pub columns: usize,
    /// The execution mode the operation resolved to.
    pub execution_mode: SKExecutionMode,
    /// The math backend the operation uses for heavy dense algebra.
    pub backend: &'static str,
}

/// A live span around a single operation, recording duration and errors.
///
/// Created with [`SKOperationObservation::begin`], used with
/// [`SKOperationObservation::in_scope`] (or [`SKOperationObservation::enter`])
/// and finalized with [`SKOperationObservation::finish`]; a failing operation
/// records its error via [`SKOperationObservation::record_error`].
pub struct SKOperationObservation {
    span: tracing::Span,
    started_at: Instant,
}

impl SKOperationObservation {
    /// Open an operation span recording the given attributes.
    pub fn begin(attributes: SKOperationAttributes) -> Self {
        let span = tracing::info_span!(
            "sciencekit.operation",
            operation = attributes.operation,
            rows = attributes.rows,
            columns = attributes.columns,
            mode = ?attributes.execution_mode,
            backend = attributes.backend,
            duration_ms = tracing::field::Empty,
            error = tracing::field::Empty,
        );
        SKOperationObservation {
            span,
            started_at: Instant::now(),
        }
    }

    /// Run `operation` inside the span's context.
    pub fn in_scope<F, T>(&self, operation: F) -> T
    where
        F: FnOnce() -> T,
    {
        self.span.in_scope(operation)
    }

    /// Enter the span manually; pair with the returned guard dropping.
    pub fn enter(&self) -> tracing::span::Entered<'_> {
        self.span.enter()
    }

    /// Record a failing operation's error on the span.
    pub fn record_error(&self, error: &dyn std::error::Error) {
        self.span.record("error", error.to_string());
    }

    /// Record the elapsed duration and close the span.
    pub fn finish(self) {
        let duration_ms = self.started_at.elapsed().as_millis() as u64;
        self.span.record("duration_ms", duration_ms);
    }
}

/// Run an operation under a [`tracing`] span, recording errors and duration.
///
/// Convenience over [`SKOperationObservation`]: algorithm crates wrap each
/// public operation so failures stay visible in the trace.
pub fn sk_run_operation<T, E>(
    attributes: SKOperationAttributes,
    operation: impl FnOnce() -> Result<T, E>,
) -> Result<T, E>
where
    E: std::error::Error,
{
    let observation = SKOperationObservation::begin(attributes);
    let result = observation.in_scope(operation);
    if let Err(error) = &result {
        observation.record_error(error);
    }
    observation.finish();
    result
}

/// Build an OpenTelemetry tracing layer over the configured global tracer.
///
/// Available only with the `observability-export` feature. The consumer
/// configures the OpenTelemetry tracer provider (e.g. an OTLP exporter) and
/// combines the returned layer with their `tracing_subscriber` registry; the
/// default build compiles this out and never initializes an exporter.
#[cfg(feature = "observability-export")]
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

#[cfg(test)]
mod observability_tests;
