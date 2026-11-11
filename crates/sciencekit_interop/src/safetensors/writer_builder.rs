//! Builder for [`super::SKSafetensorsWriter`] (spec `safetensors-format`).

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode,
    builders::{SKBuilder, SKBuilderState},
};

use super::SKSafetensorsWriter;
use super::compression::SKSafetensorsCompression;
use super::shard_plan::SKShardPlan;

const DEFAULT_HEADER_PADDING_MINIMUM: usize = 1024;

/// Builder configuring an [`SKSafetensorsWriter`].
///
/// Defaults to [`SKBackendKind::Faer`], [`SKExecutionMode::Automatic`],
/// [`SKSafetensorsCompression::None`], no sharding and zero padding.
#[derive(Debug, Clone)]
pub struct SKSafetensorsWriterBuilder {
    pub(super) state: SKBuilderState,
    pub(super) backend: SKBackendKind,
    pub(super) shard_plan: Option<SKShardPlan>,
    pub(super) pending_maximum_shard_bytes: Option<usize>,
    pub(super) header_padding_bytes: usize,
    pub(super) payload_padding_bytes: usize,
    pub(super) compression: SKSafetensorsCompression,
}

impl SKSafetensorsWriterBuilder {
    /// Start a builder with the `Faer` backend and automatic execution.
    pub fn new() -> Self {
        Self {
            state: SKBuilderState::new(),
            backend: SKBackendKind::Faer,
            shard_plan: None,
            pending_maximum_shard_bytes: None,
            header_padding_bytes: 0,
            payload_padding_bytes: 0,
            compression: SKSafetensorsCompression::None,
        }
    }

    /// Attach a pre-built [`SKShardPlan`] for splitting large model states.
    pub fn shard_plan(mut self, shard_plan: SKShardPlan) -> Self {
        self.shard_plan = Some(shard_plan);
        self.pending_maximum_shard_bytes = None;
        self
    }

    /// Configure sharding directly by specifying the maximum payload bytes per shard.
    pub fn maximum_shard_bytes(mut self, maximum_shard_bytes: usize) -> Self {
        self.pending_maximum_shard_bytes = Some(maximum_shard_bytes);
        self.shard_plan = None;
        self
    }

    /// Reserve payload padding bytes and matching header slack for in-place appends.
    pub fn padding_bytes(mut self, padding_bytes: usize) -> Self {
        self.payload_padding_bytes = padding_bytes;
        self.header_padding_bytes = if padding_bytes > 0 {
            padding_bytes.max(DEFAULT_HEADER_PADDING_MINIMUM)
        } else {
            0
        };
        self
    }

    /// Override the reserved JSON header padding slack in bytes.
    pub fn header_padding_bytes(mut self, header_padding_bytes: usize) -> Self {
        self.header_padding_bytes = header_padding_bytes;
        self
    }

    /// Override the reserved payload padding capacity in bytes.
    pub fn payload_padding_bytes(mut self, payload_padding_bytes: usize) -> Self {
        self.payload_padding_bytes = payload_padding_bytes;
        self
    }

    /// Select an optional whole-file compression wrapper.
    pub fn compression(mut self, compression: SKSafetensorsCompression) -> Self {
        self.compression = compression;
        self
    }

    /// Select the math backend recorded on writer observability spans.
    pub fn backend(mut self, backend: SKBackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Override the execution intent; defaults to [`SKExecutionMode::Automatic`].
    pub fn execution_mode<E>(mut self, mode: E) -> Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    /// Finalize and validate this builder into an [`SKSafetensorsWriter`].
    pub fn build(self) -> Result<SKSafetensorsWriter, SKError> {
        <Self as SKBuilder<SKSafetensorsWriter>>::build(self)
    }
}

impl Default for SKSafetensorsWriterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SKBuilder<SKSafetensorsWriter> for SKSafetensorsWriterBuilder {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKSafetensorsWriter, SKError> {
        let resolved_shard_plan = match (self.shard_plan, self.pending_maximum_shard_bytes) {
            (Some(plan), _) => Some(plan),
            (None, Some(threshold)) => Some(
                SKShardPlan::builder()
                    .maximum_shard_bytes(threshold)
                    .backend(self.backend)
                    .execution_mode(self.state.execution_intent())
                    .build()?,
            ),
            (None, None) => None,
        };
        Ok(SKSafetensorsWriter {
            plan: self.state,
            backend: self.backend,
            shard_plan: resolved_shard_plan,
            header_padding_bytes: self.header_padding_bytes,
            payload_padding_bytes: self.payload_padding_bytes,
            compression: self.compression,
        })
    }
}
