//! Builder for [`super::SKSafetensorsWriter`] (spec `safetensors-format`).

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode,
    builders::{SKBuilder, SKBuilderState},
};

use super::SKSafetensorsWriter;

/// Builder configuring an [`SKSafetensorsWriter`].
///
/// Defaults to [`SKBackendKind::Faer`] and [`SKExecutionMode::Automatic`].
#[derive(Debug, Clone)]
pub struct SKSafetensorsWriterBuilder {
    pub(super) state: SKBuilderState,
    pub(super) backend: SKBackendKind,
}

impl SKSafetensorsWriterBuilder {
    /// Start a builder with the `Faer` backend and automatic execution.
    pub fn new() -> Self {
        Self {
            state: SKBuilderState::new(),
            backend: SKBackendKind::Faer,
        }
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
        Ok(SKSafetensorsWriter {
            plan: self.state,
            backend: self.backend,
        })
    }
}
