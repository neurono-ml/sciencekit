//! Extended Safetensors writer over [`sciencekit_common::SKModelState`]
//! (spec `safetensors-format`).

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKExportable, SKFloat, SKModelState,
    builders::{SKBuilder, SKBuilderState},
};

use super::header::SK_SAFETENSORS_FORMAT_VERSION;
use super::single_file::encode_single_file_bytes;
use super::writer_builder::SKSafetensorsWriterBuilder;

/// Serializes fitted model states into the extended Safetensors binary format.
///
/// Construct exclusively via [`SKSafetensorsWriter::builder`].
#[derive(Debug, Clone)]
pub struct SKSafetensorsWriter {
    pub(super) plan: SKBuilderState,
    pub(super) backend: SKBackendKind,
}

impl SKSafetensorsWriter {
    /// Start configuring a Safetensors writer.
    pub fn builder() -> SKSafetensorsWriterBuilder {
        SKSafetensorsWriterBuilder::new()
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
            "sciencekit.interop.safetensors.write",
            format_version = SK_SAFETENSORS_FORMAT_VERSION,
            estimator = state.metadata().estimator_name(),
            tensor_count = state.number_of_tensors(),
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

    /// Serialize `state` into an in-memory Safetensors byte vector.
    pub fn write_state_to_bytes<F: SKFloat>(
        &self,
        state: &SKModelState<F>,
    ) -> Result<Vec<u8>, SKError> {
        self.run_write_span(state, || encode_single_file_bytes(state))
    }

    /// Write `state` in Safetensors format into `writer`.
    pub fn write_state_to_writer<F: SKFloat, W: io::Write>(
        &self,
        state: &SKModelState<F>,
        mut writer: W,
    ) -> Result<(), SKError> {
        self.run_write_span(state, || {
            let encoded = encode_single_file_bytes(state)?;
            writer.write_all(&encoded)?;
            Ok(())
        })
    }

    /// Write `state` in Safetensors format to `path`.
    pub fn write_state_to_path<F: SKFloat>(
        &self,
        state: &SKModelState<F>,
        path: &Path,
    ) -> Result<(), SKError> {
        self.run_write_span(state, || {
            let encoded = encode_single_file_bytes(state)?;
            let file = File::create(path)?;
            let mut buffered = BufWriter::new(file);
            buffered.write_all(&encoded)?;
            buffered.flush()?;
            Ok(())
        })
    }

    /// Export `model` through [`SKExportable`] and serialize its state into bytes.
    pub fn export_to_bytes<F: SKFloat, M: SKExportable<F>>(
        &self,
        model: &M,
    ) -> Result<Vec<u8>, SKError> {
        let state = model.export_state()?;
        self.write_state_to_bytes(&state)
    }

    /// Export `model` through [`SKExportable`] and write its state to `path`.
    pub fn export_to_path<F: SKFloat, M: SKExportable<F>>(
        &self,
        model: &M,
        path: &Path,
    ) -> Result<(), SKError> {
        let state = model.export_state()?;
        self.write_state_to_path(&state, path)
    }
}

/// Serialize `state` into standard Safetensors bytes with default configuration.
pub fn sk_write_model_state_safetensors<F: SKFloat>(
    state: &SKModelState<F>,
) -> Result<Vec<u8>, SKError> {
    SKSafetensorsWriter::builder()
        .build()?
        .write_state_to_bytes(state)
}

/// Write `state` as a Safetensors file to `path` with default configuration.
pub fn sk_write_model_state_safetensors_file<F: SKFloat>(
    path: &Path,
    state: &SKModelState<F>,
) -> Result<(), SKError> {
    SKSafetensorsWriter::builder()
        .build()?
        .write_state_to_path(state, path)
}

/// Export `model` via [`SKExportable`] into standard Safetensors bytes.
pub fn sk_export_model_safetensors<F: SKFloat, M: SKExportable<F>>(
    model: &M,
) -> Result<Vec<u8>, SKError> {
    SKSafetensorsWriter::builder()
        .build()?
        .export_to_bytes(model)
}

/// Export `model` via [`SKExportable`] into a Safetensors file at `path`.
pub fn sk_export_model_safetensors_file<F: SKFloat, M: SKExportable<F>>(
    path: &Path,
    model: &M,
) -> Result<(), SKError> {
    SKSafetensorsWriter::builder()
        .build()?
        .export_to_path(model, path)
}
