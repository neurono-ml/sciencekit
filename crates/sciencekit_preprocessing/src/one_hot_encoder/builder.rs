//! The `SKOneHotEncoder` builder (spec `one-hot-encoder`).

use sciencekit_common::{SKBackendKind, SKBuilder, SKBuilderState, SKError, SKExecutionMode};

use crate::one_hot_encoder::core_implementation::SKOneHotEncoder;
use crate::one_hot_encoder::unknown_policy::SKUnknownPolicy;

/// Builder for [`SKOneHotEncoder`]; the only construction path.
pub struct SKOneHotEncoderBuilder {
    pub(super) state: SKBuilderState,
    pub(super) sparse_output: bool,
    pub(super) handle_unknown: SKUnknownPolicy,
    pub(super) backend: SKBackendKind,
}

impl SKOneHotEncoderBuilder {
    /// A builder with the documented defaults.
    pub fn new() -> SKOneHotEncoderBuilder {
        Self {
            state: SKBuilderState::new(),
            sparse_output: true,
            handle_unknown: SKUnknownPolicy::Reject,
            backend: SKBackendKind::Faer,
        }
    }

    /// Whether the default transform returns a sparse CSR matrix.
    pub fn sparse_output(mut self, value: bool) -> Self {
        self.sparse_output = value;
        self
    }

    /// The unknown-category policy.
    pub fn handle_unknown(mut self, policy: SKUnknownPolicy) -> Self {
        self.handle_unknown = policy;
        self
    }

    /// The math backend kind recorded in operation spans and carried into the
    /// fitted model.
    pub fn backend(mut self, backend: SKBackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// The execution intent.
    pub fn execution_mode<E>(mut self, mode: E) -> Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }
}

impl Default for SKOneHotEncoderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SKBuilder<SKOneHotEncoder> for SKOneHotEncoderBuilder {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKOneHotEncoder, SKError> {
        Ok(SKOneHotEncoder {
            state: self.state,
            sparse_output: self.sparse_output,
            handle_unknown: self.handle_unknown,
            backend: self.backend,
        })
    }
}
