//! Builder for [`super::SKSafetensorsWriter`] (spec `safetensors-format`).

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode,
    builders::{SKBuilder, SKBuilderState},
};

use super::SKSafetensorsWriter;
use super::shard_plan::SKShardPlan;

/// Builder configuring an [`SKSafetensorsWriter`].
///
/// Defaults to [`SKBackendKind::Faer`] and [`SKExecutionMode::Automatic`].
#[derive(Debug, Clone)]
pub struct SKSafetensorsWriterBuilder {
    pub(super) state: SKBuilderState,
    pub(super) backend: SKBackendKind,
    pub(super) shard_plan: Option<SKShardPlan>,
    pub(super) pending_maximum_shard_bytes: Option<usize>,
}

impl SKSafetensorsWriterBuilder {
    /// Start a builder with the `Faer` backend and automatic execution.
    pub fn new() -> Self {
        Self {
            state: SKBuilderState::new(),
            backend: SKBackendKind::Faer,
            shard_plan: None,
            pending_maximum_shard_bytes: None,
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
        })
    }
}
