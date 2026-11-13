//! Observability for OpenCL operations (spec `observability`, PRD §9.2).
//!
//! Every operation opens a span carrying the configured backend kind
//! (`opencl`, never a host kind), the operation name, the device index and the
//! bytes moved. Recorded this way, a trace answers *where* the work ran and
//! *how much* data crossed the bus without the caller adding instrumentation.

use sciencekit_common::SKBackendKind;

/// The tracing target every OpenCL span uses.
pub(crate) const OPENCL_TRACING_TARGET: &str = "sciencekit_gpu_opencl";

/// Open the span for one operation.
pub(crate) fn operation_span(operation: &'static str) -> tracing::Span {
    tracing::info_span!(
        target: OPENCL_TRACING_TARGET,
        "opencl_operation",
        backend = %SKBackendKind::OpenCl,
        operation = %operation,
        device_index = tracing::field::Empty,
        transfer_bytes = tracing::field::Empty,
    )
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
