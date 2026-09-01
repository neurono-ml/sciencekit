//! Live operation span observation (spec `observability`).

use std::time::Instant;

use super::operation_attributes::SKOperationAttributes;

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
            operation = %attributes.operation,
            rows = attributes.rows,
            columns = attributes.columns,
            mode = ?attributes.execution_mode,
            backend = %attributes.backend,
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
