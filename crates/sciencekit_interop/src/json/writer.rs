//! Human-readable JSON debug writer over [`sciencekit_common::SKModelState`]
//! (spec `json-debug-format`).

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKExportable, SKFloat, SKModelState,
    builders::{SKBuilder, SKBuilderState},
};

use super::schema::{encode_model_state_to_json_string, encode_model_state_to_writer};
use super::{SK_JSON_DEBUG_FORMAT_VERSION, SKJsonDebugWriterBuilder, SKJsonFormatMode};

/// Serializes fitted model states into deterministic, human-readable JSON.
///
/// Construct via [`SKJsonDebugWriter::builder`]. Pretty formatting is the
/// default so exported files diff cleanly line by line; compact formatting
/// omits insignificant whitespace.
#[derive(Debug, Clone, Copy)]
pub struct SKJsonDebugWriter {
    pub(super) plan: SKBuilderState,
    pub(super) format_mode: SKJsonFormatMode,
    pub(super) backend: SKBackendKind,
}

impl SKJsonDebugWriter {
    /// Start configuring a JSON debug writer.
    pub fn builder() -> SKJsonDebugWriterBuilder {
        SKJsonDebugWriterBuilder::new()
    }

    /// The configured output whitespace mode.
    pub fn format_mode(&self) -> SKJsonFormatMode {
        self.format_mode
    }

    /// The configured math backend kind.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// The configured execution intent.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.plan.execution_intent()
    }

    fn run_write_span<F: SKFloat, T>(
        &self,
        state: &SKModelState<F>,
        operation: impl FnOnce() -> Result<T, SKError>,
    ) -> Result<T, SKError> {
        let span = tracing::info_span!(
            "sciencekit.interop.json.write",
            format_version = SK_JSON_DEBUG_FORMAT_VERSION,
            estimator = state.metadata().estimator_name(),
            tensor_count = state.number_of_tensors(),
            format_mode = %self.format_mode,
            backend = %self.backend,
            execution_mode = ?self.plan.execution_intent(),
            duration_ms = tracing::field::Empty,
            error = tracing::field::Empty,
        );
        let started_at = Instant::now();
        let result = span.in_scope(operation);
        if let Err(error) = &result {
            span.record("error", error.to_string());
        }
        span.record("duration_ms", started_at.elapsed().as_millis() as u64);
        result
    }

    /// Serialize `state` into a UTF-8 JSON string.
    pub fn write_state_to_string<F: SKFloat>(
        &self,
        state: &SKModelState<F>,
    ) -> Result<String, SKError> {
        self.run_write_span(state, || {
            encode_model_state_to_json_string(state, self.format_mode)
        })
    }

    /// Stream `state` as UTF-8 JSON into `writer`.
    pub fn write_state_to_writer<F: SKFloat, W: io::Write>(
        &self,
        state: &SKModelState<F>,
        writer: W,
    ) -> Result<(), SKError> {
        self.run_write_span(state, || {
            encode_model_state_to_writer(state, self.format_mode, writer)
        })
    }

    /// Write `state` as UTF-8 JSON to `path`.
    pub fn write_state_to_path<F: SKFloat>(
        &self,
        state: &SKModelState<F>,
        path: &Path,
    ) -> Result<(), SKError> {
        self.run_write_span(state, || {
            let file = File::create(path)?;
            let mut buffered = BufWriter::new(file);
            encode_model_state_to_writer(state, self.format_mode, &mut buffered)?;
            buffered.flush()?;
            Ok(())
        })
    }

    /// Export a fitted model through [`SKExportable`] and serialize it to a JSON string.
    pub fn export_to_string<F: SKFloat, M: SKExportable<F>>(
        &self,
        model: &M,
    ) -> Result<String, SKError> {
        let state = model.export_state()?;
        self.write_state_to_string(&state)
    }

    /// Export a fitted model through [`SKExportable`] and write it to `path`.
    pub fn export_to_path<F: SKFloat, M: SKExportable<F>>(
        &self,
        model: &M,
        path: &Path,
    ) -> Result<(), SKError> {
        let state = model.export_state()?;
        self.write_state_to_path(&state, path)
    }
}

/// Serialize `state` into JSON with the requested [`SKJsonFormatMode`].
pub fn sk_write_model_state_json<F: SKFloat>(
    state: &SKModelState<F>,
    format_mode: SKJsonFormatMode,
) -> Result<String, SKError> {
    SKJsonDebugWriter::builder()
        .format_mode(format_mode)
        .build()?
        .write_state_to_string(state)
}

/// Serialize `state` into pretty-printed, human-readable JSON.
pub fn sk_write_model_state_json_pretty<F: SKFloat>(
    state: &SKModelState<F>,
) -> Result<String, SKError> {
    sk_write_model_state_json(state, SKJsonFormatMode::Pretty)
}

/// Serialize `state` into single-line compact JSON.
pub fn sk_write_model_state_json_compact<F: SKFloat>(
    state: &SKModelState<F>,
) -> Result<String, SKError> {
    sk_write_model_state_json(state, SKJsonFormatMode::Compact)
}

/// Write `state` as JSON to `path` with the requested [`SKJsonFormatMode`].
pub fn sk_write_model_state_json_file<F: SKFloat>(
    path: &Path,
    state: &SKModelState<F>,
    format_mode: SKJsonFormatMode,
) -> Result<(), SKError> {
    SKJsonDebugWriter::builder()
        .format_mode(format_mode)
        .build()?
        .write_state_to_path(state, path)
}

/// Export `model` via [`SKExportable`] and serialize its state into JSON.
pub fn sk_export_model_json<F: SKFloat, M: SKExportable<F>>(
    model: &M,
    format_mode: SKJsonFormatMode,
) -> Result<String, SKError> {
    SKJsonDebugWriter::builder()
        .format_mode(format_mode)
        .build()?
        .export_to_string(model)
}
