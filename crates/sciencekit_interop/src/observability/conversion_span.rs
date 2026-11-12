//! The interop conversion span (spec `dataset-interop-core`).

use std::time::Instant;

use sciencekit_common::SKError;

use super::SKInteropSourceKind;
use crate::missing::SKMissingValuePolicy;

/// Structured attributes recorded on an interop conversion span.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SKInteropConversionAttributes {
    /// The engine the converted batch came from.
    pub source_kind: SKInteropSourceKind,
    /// The number of rows in the converted batch.
    pub rows_count: usize,
    /// The number of selected columns.
    pub columns_count: usize,
    /// The missing-value policy applied.
    pub missing_value_policy: SKMissingValuePolicy,
}

/// Run a tabular conversion under a `tracing` span, recording duration and
/// errors.
///
/// The span carries the source kind, the batch shape and the missing-value
/// policy, so an operator can tell which engine produced a block and why a
/// conversion failed.
pub fn sk_run_interop_conversion<T>(
    attributes: SKInteropConversionAttributes,
    conversion: impl FnOnce() -> Result<T, SKError>,
) -> Result<T, SKError> {
    let span = tracing::info_span!(
        "sciencekit.interop.conversion",
        source_kind = %attributes.source_kind,
        rows = attributes.rows_count,
        columns = attributes.columns_count,
        policy = %attributes.missing_value_policy,
        duration_ms = tracing::field::Empty,
        error = tracing::field::Empty,
    );
    let started_at = Instant::now();
    let result = span.in_scope(conversion);
    if let Err(error) = &result {
        span.record("error", error.to_string());
    }
    span.record("duration_ms", started_at.elapsed().as_millis() as u64);
    result
}
