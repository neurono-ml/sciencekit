//! Live operation span observation (spec `observability`, spec
//! `gpu-compute-contracts`).

use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::device::{SKFallbackReason, sk_resolve_operation_backend};

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
    fallback_recorded: AtomicBool,
}

impl SKOperationObservation {
    /// Open an operation span recording the resolved backend and any fallback
    /// reason.
    pub fn begin(attributes: SKOperationAttributes) -> Self {
        let (backend, fallback_reason) = sk_resolve_operation_backend(attributes.backend);
        let span = tracing::info_span!(
            "sciencekit.operation",
            operation = %attributes.operation,
            rows = attributes.rows_count,
            columns = attributes.columns_count,
            mode = ?attributes.execution_mode,
            backend = %backend,
            fallback_reason = tracing::field::Empty,
            duration_ms = tracing::field::Empty,
            error = tracing::field::Empty,
        );
        let observation = SKOperationObservation {
            span,
            started_at: Instant::now(),
            fallback_recorded: AtomicBool::new(false),
        };
        if let Some(reason) = fallback_reason {
            observation.record_fallback_reason(reason);
        }
        observation
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

    /// Record why the operation ran on the host backend instead of a device.
    ///
    /// Idempotent per span: if [`SKOperationObservation::begin`] or an earlier
    /// call already recorded a fallback reason, subsequent calls do not record
    /// duplicate field entries.
    pub fn record_fallback_reason(&self, reason: SKFallbackReason) {
        if !self.fallback_recorded.swap(true, Ordering::AcqRel) {
            self.span
                .record("fallback_reason", tracing::field::display(reason));
        }
    }

    /// Record the elapsed duration and close the span.
    pub fn finish(self) {
        let duration_ms = self.started_at.elapsed().as_millis() as u64;
        self.span.record("duration_ms", duration_ms);
    }
}
