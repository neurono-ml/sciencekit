//! Observability for OpenCL operations (spec `observability`, PRD §9.2).
//!
//! Every operation opens a span carrying the backend kind the operation
//! *actually* resolves to, the operation name, the device index and the bytes
//! moved. When a device serves the call the kind is
//! [`SKBackendKind::OpenCl`](sciencekit_common::SKBackendKind::OpenCl); when no
//! device can serve it the span records the host fallback kind and the
//! `SKFallbackReason` instead, matching
//! [`sk_resolve_operation_backend`](sciencekit_common::sk_resolve_operation_backend).
//! A trace therefore never reports OpenCL for work that ran (or fell back) on
//! the CPU, and the caller sees *why* the device was not used.

use sciencekit_common::{SKBackendKind, SKFallbackReason};

/// The tracing target every OpenCL span uses.
pub(crate) const OPENCL_TRACING_TARGET: &str = "sciencekit_gpu_opencl";

/// Open the span for one operation.
///
/// `backend` is the kind the operation resolves to and `fallback_reason` is
/// `Some` exactly when that kind is the host fallback.
pub(crate) fn operation_span(
    operation: &'static str,
    backend: SKBackendKind,
    fallback_reason: Option<SKFallbackReason>,
) -> tracing::Span {
    let span = tracing::info_span!(
        target: OPENCL_TRACING_TARGET,
        "opencl_operation",
        backend = %backend,
        operation = %operation,
        fallback_reason = tracing::field::Empty,
        device_index = tracing::field::Empty,
        transfer_bytes = tracing::field::Empty,
    );
    if let Some(reason) = fallback_reason {
        span.record("fallback_reason", tracing::field::display(reason));
    }
    span
}

/// Record the device the operation ran on.
#[cfg(feature = "opencl")]
pub(crate) fn record_device(span: &tracing::Span, device_index: usize) {
    span.record("device_index", device_index as u64);
}

/// Record the host↔device bytes the operation moved.
#[cfg(feature = "opencl")]
pub(crate) fn record_transfer(span: &tracing::Span, bytes: u64) {
    span.record("transfer_bytes", bytes);
}
