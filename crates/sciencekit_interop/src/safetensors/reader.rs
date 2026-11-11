//! Extended Safetensors reader reconstructing [`sciencekit_common::SKModelState`]
//! (spec `safetensors-format`).

use std::fs::File;
use std::io::{self, BufReader, Read};
use std::path::Path;
use std::time::Instant;

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKFloat, SKImportable, SKModelState,
    builders::SKBuilderState,
};

use super::compression::decompress_safetensors_bytes;
use super::header::SK_SAFETENSORS_FORMAT_VERSION;
use super::reader_builder::SKSafetensorsReaderBuilder;
use super::shard_index::sk_safetensors_index_path;
use super::shard_io::read_sharded_state_from_index;
use super::single_file::decode_single_file_bytes;

/// Parses and validates Safetensors files and shard sets into [`SKModelState`].
///
/// Construct exclusively via [`SKSafetensorsReader::builder`].
#[derive(Debug, Clone, Copy)]
pub struct SKSafetensorsReader {
    pub(super) plan: SKBuilderState,
    pub(super) backend: SKBackendKind,
}

impl SKSafetensorsReader {
    /// Start configuring a Safetensors reader.
    pub fn builder() -> SKSafetensorsReaderBuilder {
        SKSafetensorsReaderBuilder::new()
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
            "sciencekit.interop.safetensors.read",
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
                span.record("format_version", SK_SAFETENSORS_FORMAT_VERSION);
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

    /// Parse and validate a raw or compressed Safetensors byte slice into [`SKModelState`].
    pub fn read_state_from_bytes<F: SKFloat>(
        &self,
        bytes: &[u8],
    ) -> Result<SKModelState<F>, SKError> {
        self.run_read_span(|| {
            let uncompressed = decompress_safetensors_bytes(None, bytes)?;
            decode_single_file_bytes(&uncompressed)
        })
    }

    /// Read and validate a raw or compressed Safetensors stream into [`SKModelState`].
    pub fn read_state_from_reader<F: SKFloat, R: io::Read>(
        &self,
        mut reader: R,
    ) -> Result<SKModelState<F>, SKError> {
        self.run_read_span(|| {
            let mut buffer = Vec::new();
            reader.read_to_end(&mut buffer)?;
            let uncompressed = decompress_safetensors_bytes(None, &buffer)?;
            decode_single_file_bytes(&uncompressed)
        })
    }

    /// Read and validate a single (or compressed) Safetensors file or sharded manifest at `path`.
    pub fn read_state_from_path<F: SKFloat>(
        &self,
        path: &Path,
    ) -> Result<SKModelState<F>, SKError> {
        self.run_read_span(|| {
            if path.to_string_lossy().ends_with(".index.json") {
                return read_sharded_state_from_index(path, |shard_path, raw| {
                    decompress_safetensors_bytes(Some(shard_path), raw)
                });
            }
            let companion_index = sk_safetensors_index_path(path);
            if companion_index.exists() {
                return read_sharded_state_from_index(&companion_index, |shard_path, raw| {
                    decompress_safetensors_bytes(Some(shard_path), raw)
                });
            }
            let file = File::open(path)?;
            let mut buffer = Vec::new();
            BufReader::new(file).read_to_end(&mut buffer)?;
            let uncompressed = decompress_safetensors_bytes(Some(path), &buffer)?;
            decode_single_file_bytes(&uncompressed)
        })
    }

    /// Parse Safetensors bytes and reconstruct a model through [`SKImportable`].
    pub fn import_from_bytes<F: SKFloat, M: SKImportable<F>>(
        &self,
        bytes: &[u8],
    ) -> Result<M, SKError> {
        let state = self.read_state_from_bytes::<F>(bytes)?;
        M::import_state(&state)
    }

    /// Read a Safetensors file or shard set at `path` and reconstruct a model.
    pub fn import_from_path<F: SKFloat, M: SKImportable<F>>(
        &self,
        path: &Path,
    ) -> Result<M, SKError> {
        let state = self.read_state_from_path::<F>(path)?;
        M::import_state(&state)
    }
}

/// Parse and validate Safetensors bytes into an [`SKModelState`].
pub fn sk_read_model_state_safetensors<F: SKFloat>(
    bytes: &[u8],
) -> Result<SKModelState<F>, SKError> {
    SKSafetensorsReader::builder()
        .build()?
        .read_state_from_bytes(bytes)
}

/// Read and validate a Safetensors file or shard set at `path` into an [`SKModelState`].
pub fn sk_read_model_state_safetensors_file<F: SKFloat>(
    path: &Path,
) -> Result<SKModelState<F>, SKError> {
    SKSafetensorsReader::builder()
        .build()?
        .read_state_from_path(path)
}

/// Parse Safetensors bytes and reconstruct a model through [`SKImportable`].
pub fn sk_import_model_safetensors<F: SKFloat, M: SKImportable<F>>(
    bytes: &[u8],
) -> Result<M, SKError> {
    SKSafetensorsReader::builder()
        .build()?
        .import_from_bytes(bytes)
}

/// Read a Safetensors file or shard set at `path` and reconstruct a model through [`SKImportable`].
pub fn sk_import_model_safetensors_file<F: SKFloat, M: SKImportable<F>>(
    path: &Path,
) -> Result<M, SKError> {
    SKSafetensorsReader::builder()
        .build()?
        .import_from_path(path)
}
