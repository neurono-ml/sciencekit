//! Sharding plan, builder and shard-index value type (spec `safetensors-format`).

use std::collections::BTreeMap;

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKFloat, SKModelState, SKTensor, SKTensorDataType,
    builders::{SKBuilder, SKBuilderState},
};

/// Default byte threshold per shard (`512 MiB`).
pub const SK_DEFAULT_MAXIMUM_SHARD_BYTES: usize = 512 * 1024 * 1024;

/// Immutable shard-splitting configuration over [`SKModelState`].
///
/// Construct exclusively via [`SKShardPlan::builder`].
#[derive(Debug, Clone, Copy)]
pub struct SKShardPlan {
    plan: SKBuilderState,
    maximum_shard_bytes: usize,
    backend: SKBackendKind,
}

impl SKShardPlan {
    /// Start configuring an [`SKShardPlan`].
    pub fn builder() -> SKShardPlanBuilder {
        SKShardPlanBuilder::new()
    }

    /// Maximum tensor payload bytes placed in a single shard when multiple tensors share it.
    pub fn maximum_shard_bytes(&self) -> usize {
        self.maximum_shard_bytes
    }

    /// The configured math backend kind.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// The configured execution intent.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.plan.execution_intent()
    }

    /// Partition `state` tensors deterministically by name into shard groups.
    pub(crate) fn partition_state<'state, F: SKFloat>(
        &self,
        state: &'state SKModelState<F>,
    ) -> Vec<Vec<&'state SKTensor<F>>> {
        let mut groups: Vec<Vec<&'state SKTensor<F>>> = Vec::new();
        let mut current_group: Vec<&'state SKTensor<F>> = Vec::new();
        let mut current_bytes = 0_usize;

        for tensor in state.tensors().values() {
            let tensor_bytes = tensor_payload_bytes(tensor);
            if !current_group.is_empty()
                && current_bytes.saturating_add(tensor_bytes) > self.maximum_shard_bytes
            {
                groups.push(current_group);
                current_group = Vec::new();
                current_bytes = 0;
            }
            current_group.push(tensor);
            current_bytes = current_bytes.saturating_add(tensor_bytes);
        }

        if !current_group.is_empty() || groups.is_empty() {
            groups.push(current_group);
        }
        groups
    }
}

/// Builder configuring an [`SKShardPlan`].
#[derive(Debug, Clone, Copy)]
pub struct SKShardPlanBuilder {
    state: SKBuilderState,
    maximum_shard_bytes: usize,
    backend: SKBackendKind,
}

impl SKShardPlanBuilder {
    /// Create a shard plan builder defaulting to `512 MiB`, `Faer` and `Automatic`.
    pub fn new() -> Self {
        Self {
            state: SKBuilderState::new(),
            maximum_shard_bytes: SK_DEFAULT_MAXIMUM_SHARD_BYTES,
            backend: SKBackendKind::Faer,
        }
    }

    /// Set the per-shard byte threshold (must be greater than zero).
    pub fn maximum_shard_bytes(mut self, maximum_shard_bytes: usize) -> Self {
        self.maximum_shard_bytes = maximum_shard_bytes;
        self
    }

    /// Select the math backend recorded on spans.
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

    /// Validate and build an [`SKShardPlan`].
    pub fn build(self) -> Result<SKShardPlan, SKError> {
        <Self as SKBuilder<SKShardPlan>>::build(self)
    }
}

impl Default for SKShardPlanBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SKBuilder<SKShardPlan> for SKShardPlanBuilder {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKShardPlan, SKError> {
        if self.maximum_shard_bytes == 0 {
            return Err(SKError::InvalidHyperparameter {
                name: "maximum_shard_bytes",
                reason: "maximum_shard_bytes must be greater than zero".to_string(),
            });
        }
        Ok(SKShardPlan {
            plan: self.state,
            maximum_shard_bytes: self.maximum_shard_bytes,
            backend: self.backend,
        })
    }
}

/// Validated view of a `.safetensors.index.json` shard manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SKShardIndex {
    pub(crate) format_version: u32,
    pub(crate) metadata: BTreeMap<String, String>,
    pub(crate) shards: Vec<String>,
    pub(crate) tensor_to_shard: BTreeMap<String, String>,
}

impl SKShardIndex {
    /// Container major format version recorded in the index.
    pub fn format_version(&self) -> u32 {
        self.format_version
    }

    /// Raw string metadata dictionary carried by the index.
    pub fn metadata(&self) -> &BTreeMap<String, String> {
        &self.metadata
    }

    /// Ordered relative shard filenames belonging to this model state.
    pub fn shards(&self) -> &[String] {
        &self.shards
    }

    /// Deterministic map from tensor name to relative shard filename.
    pub fn tensor_to_shard(&self) -> &BTreeMap<String, String> {
        &self.tensor_to_shard
    }
}

/// Compute the uncompressed payload byte size of `tensor`.
pub(crate) fn tensor_payload_bytes<F: SKFloat>(tensor: &SKTensor<F>) -> usize {
    let element_width = match tensor.data_type() {
        SKTensorDataType::F32 => 4,
        SKTensorDataType::F64 | SKTensorDataType::I64 => 8,
        SKTensorDataType::Bool => 1,
        _ => 8,
    };
    tensor.number_of_elements().saturating_mul(element_width)
}

/// Compute the total uncompressed tensor payload bytes of `state`.
pub(crate) fn state_total_payload_bytes<F: SKFloat>(state: &SKModelState<F>) -> usize {
    state
        .tensors()
        .values()
        .map(tensor_payload_bytes)
        .fold(0_usize, usize::saturating_add)
}
