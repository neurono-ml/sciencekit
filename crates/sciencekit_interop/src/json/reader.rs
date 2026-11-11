//! Human-readable JSON debug reader reconstructing [`sciencekit_common::SKModelState`]
//! (spec `json-debug-format`).

use std::fs::File;
use std::io::{self, BufReader};
use std::path::Path;
use std::time::Instant;

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKFloat, SKImportable, SKModelState,
    builders::{SKBuilder, SKBuilderState},
};

use super::SKJsonDebugReaderBuilder;
use super::schema_deserialization::RawJsonModelDocument;
use super::state_decoding::decode_raw_document;

/// Parses and validates human-readable JSON debug documents into [`SKModelState`].
///
/// Construct via [`SKJsonDebugReader::builder`]. Unknown fields within major
/// version `1` are ignored for forward compatibility.
#[derive(Debug, Clone, Copy)]
pub struct SKJsonDebugReader {
    pub(super) plan: SKBuilderState,
    pub(super) backend: SKBackendKind,
}

impl SKJsonDebugReader {
    /// Start configuring a JSON debug reader.
    pub fn builder() -> SKJsonDebugReaderBuilder {
        SKJsonDebugReaderBuilder::new()
    }

    /// The configured math backend kind.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// The configured execution intent.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.plan.execution_intent()
    }

    fn run_read_span<F: SKFloat>(
        &self,
        operation: impl FnOnce() -> Result<SKModelState<F>, SKError>,
    ) -> Result<SKModelState<F>, SKError> {
        let span = tracing::info_span!(
            "sciencekit.interop.json.read",
            format_version = tracing::field::Empty,
            estimator = tracing::field::Empty,
            tensor_count = tracing::field::Empty,
            backend = %self.backend,
            execution_mode = ?self.plan.execution_intent(),
            duration_ms = tracing::field::Empty,
            error = tracing::field::Empty,
        );
        let started_at = Instant::now();
        let result = span.in_scope(operation);
        match &result {
            Ok(state) => {
                span.record("format_version", state.metadata().format_version());
                span.record("estimator", state.metadata().estimator_name());
                span.record("tensor_count", state.number_of_tensors());
            }
            Err(error) => {
                span.record("error", error.to_string());
            }
        }
        span.record("duration_ms", started_at.elapsed().as_millis() as u64);
        result
    }

    /// Parse and validate a UTF-8 JSON string into an [`SKModelState`].
    pub fn read_state_from_string<F: SKFloat>(
        &self,
        input: &str,
    ) -> Result<SKModelState<F>, SKError> {
        self.run_read_span(|| {
            let raw: RawJsonModelDocument = serde_json::from_str(input).map_err(|error| {
                SKError::Conversion(format!("invalid JSON model state: {error}"))
            })?;
            decode_raw_document(raw)
        })
    }

    /// Parse and validate UTF-8 JSON from an [`io::Read`] stream into an [`SKModelState`].
    pub fn read_state_from_reader<F: SKFloat, R: io::Read>(
        &self,
        reader: R,
    ) -> Result<SKModelState<F>, SKError> {
        self.run_read_span(|| {
            let raw: RawJsonModelDocument = serde_json::from_reader(reader).map_err(|error| {
                if error.is_io() {
                    SKError::Io(io::Error::other(error))
                } else {
                    SKError::Conversion(format!("invalid JSON model state: {error}"))
                }
            })?;
            decode_raw_document(raw)
        })
    }

    /// Read, parse and validate a JSON debug file at `path`.
    pub fn read_state_from_path<F: SKFloat>(
        &self,
        path: &Path,
    ) -> Result<SKModelState<F>, SKError> {
        let file = File::open(path)?;
        self.read_state_from_reader(BufReader::new(file))
    }

    /// Parse JSON and reconstruct a fitted model through [`SKImportable`].
    pub fn import_from_string<F: SKFloat, M: SKImportable<F>>(
        &self,
        input: &str,
    ) -> Result<M, SKError> {
        let state = self.read_state_from_string::<F>(input)?;
        M::import_state(&state)
    }

    /// Read JSON from `path` and reconstruct a fitted model through [`SKImportable`].
    pub fn import_from_path<F: SKFloat, M: SKImportable<F>>(
        &self,
        path: &Path,
    ) -> Result<M, SKError> {
        let state = self.read_state_from_path::<F>(path)?;
        M::import_state(&state)
    }
}

/// Parse and validate a JSON debug string into an [`SKModelState`].
pub fn sk_read_model_state_json<F: SKFloat>(input: &str) -> Result<SKModelState<F>, SKError> {
    SKJsonDebugReader::builder()
        .build()?
        .read_state_from_string(input)
}

/// Read, parse and validate a JSON debug file at `path` into an [`SKModelState`].
pub fn sk_read_model_state_json_file<F: SKFloat>(path: &Path) -> Result<SKModelState<F>, SKError> {
    SKJsonDebugReader::builder()
        .build()?
        .read_state_from_path(path)
}

/// Parse a JSON debug string and reconstruct a model through [`SKImportable`].
pub fn sk_import_model_json<F: SKFloat, M: SKImportable<F>>(input: &str) -> Result<M, SKError> {
    SKJsonDebugReader::builder()
        .build()?
        .import_from_string(input)
}
