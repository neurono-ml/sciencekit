//! Extended Safetensors writer over [`sciencekit_common::SKModelState`]
//! (spec `safetensors-format`).

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::time::Instant;

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKExportable, SKFloat, SKModelState, SKTensor,
    builders::SKBuilderState,
};

use super::compression::{SKSafetensorsCompression, compress_safetensors_bytes};
use super::header::SK_SAFETENSORS_FORMAT_VERSION;
use super::padding_append::append_tensors_to_path;
use super::padding_encode::{SKAppendOutcome, encode_state_with_padding};
use super::shard_io::write_sharded_state_to_path;
use super::shard_plan::{SKShardPlan, state_total_payload_bytes};
use super::writer_builder::SKSafetensorsWriterBuilder;

/// Serializes fitted model states into the extended Safetensors binary format.
///
/// Construct exclusively via [`SKSafetensorsWriter::builder`].
#[derive(Debug, Clone)]
pub struct SKSafetensorsWriter {
    pub(super) plan: SKBuilderState,
    pub(super) backend: SKBackendKind,
    pub(super) shard_plan: Option<SKShardPlan>,
    pub(super) header_padding_bytes: usize,
    pub(super) payload_padding_bytes: usize,
    pub(super) compression: SKSafetensorsCompression,
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

    /// The optional shard-splitting plan.
    pub fn shard_plan(&self) -> Option<&SKShardPlan> {
        self.shard_plan.as_ref()
    }

    /// Reserved JSON header padding bytes configured on this writer.
    pub fn header_padding_bytes(&self) -> usize {
        self.header_padding_bytes
    }

    /// Reserved payload padding bytes configured on this writer.
    pub fn payload_padding_bytes(&self) -> usize {
        self.payload_padding_bytes
    }

    /// Configured whole-file compression mode.
    pub fn compression(&self) -> SKSafetensorsCompression {
        self.compression
    }

    fn run_write_span<T>(
        &self,
        estimator: &str,
        tensor_count: usize,
        operation: impl FnOnce() -> Result<T, SKError>,
    ) -> Result<T, SKError> {
        let span = tracing::info_span!(
            "sciencekit.interop.safetensors.write",
            format_version = SK_SAFETENSORS_FORMAT_VERSION,
            estimator = estimator,
            tensor_count = tensor_count,
            compression = %self.compression,
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

    fn encode_single<F: SKFloat>(&self, state: &SKModelState<F>) -> Result<Vec<u8>, SKError> {
        let raw = encode_state_with_padding(
            state,
            self.header_padding_bytes,
            self.payload_padding_bytes,
        )?;
        compress_safetensors_bytes(&raw, self.compression)
    }

    /// Serialize `state` into an in-memory Safetensors byte vector.
    pub fn write_state_to_bytes<F: SKFloat>(
        &self,
        state: &SKModelState<F>,
    ) -> Result<Vec<u8>, SKError> {
        self.run_write_span(
            state.metadata().estimator_name(),
            state.number_of_tensors(),
            || self.encode_single(state),
        )
    }

    /// Write `state` in Safetensors format into `writer`.
    pub fn write_state_to_writer<F: SKFloat, W: io::Write>(
        &self,
        state: &SKModelState<F>,
        mut writer: W,
    ) -> Result<(), SKError> {
        let bytes = self.write_state_to_bytes(state)?;
        writer.write_all(&bytes)?;
        Ok(())
    }

    /// Write `state` to `path`, splitting into immutable shards when `shard_plan` is exceeded.
    pub fn write_state_to_path<F: SKFloat>(
        &self,
        state: &SKModelState<F>,
        path: &Path,
    ) -> Result<(), SKError> {
        self.run_write_span(
            state.metadata().estimator_name(),
            state.number_of_tensors(),
            || {
                if let Some(plan) = &self.shard_plan {
                    if state_total_payload_bytes(state) > plan.maximum_shard_bytes() {
                        return write_sharded_state_to_path(
                            state,
                            path,
                            plan,
                            self.compression.file_extension(),
                            |raw| compress_safetensors_bytes(raw, self.compression),
                        );
                    }
                }
                let encoded = self.encode_single(state)?;
                let mut buffered = BufWriter::new(File::create(path)?);
                buffered.write_all(&encoded)?;
                buffered.flush()?;
                Ok(())
            },
        )
    }

    /// Append a single `tensor` in place when padding allows, or spill to a new shard.
    pub fn append_tensor_to_path<F: SKFloat>(
        &self,
        path: &Path,
        tensor: &SKTensor<F>,
    ) -> Result<SKAppendOutcome, SKError> {
        self.append_tensors_to_path(path, std::slice::from_ref(tensor))
    }

    /// Append `tensors` in place when padding allows, or spill to a new shard.
    pub fn append_tensors_to_path<F: SKFloat>(
        &self,
        path: &Path,
        tensors: &[SKTensor<F>],
    ) -> Result<SKAppendOutcome, SKError> {
        self.run_write_span("append", tensors.len(), || {
            append_tensors_to_path(path, tensors, self.compression.file_extension(), |raw| {
                compress_safetensors_bytes(raw, self.compression)
            })
        })
    }

    /// Export `model` through [`SKExportable`] and serialize its state into bytes.
    pub fn export_to_bytes<F: SKFloat, M: SKExportable<F>>(
        &self,
        model: &M,
    ) -> Result<Vec<u8>, SKError> {
        self.write_state_to_bytes(&model.export_state()?)
    }

    /// Export `model` through [`SKExportable`] and write its state to `path`.
    pub fn export_to_path<F: SKFloat, M: SKExportable<F>>(
        &self,
        model: &M,
        path: &Path,
    ) -> Result<(), SKError> {
        self.write_state_to_path(&model.export_state()?, path)
    }
}
