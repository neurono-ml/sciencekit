//! Builder for [`super::SKSafetensorsReader`] (spec `safetensors-format`).

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode,
    builders::{SKBuilder, SKBuilderState},
};

use super::SKSafetensorsReader;

/// Builder configuring an [`SKSafetensorsReader`].
///
/// Defaults to [`SKBackendKind::Faer`] and [`SKExecutionMode::Automatic`].
#[derive(Debug, Clone, Copy)]
pub struct SKSafetensorsReaderBuilder {
    pub(super) state: SKBuilderState,
    pub(super) backend: SKBackendKind,
}

impl SKSafetensorsReaderBuilder {
    /// Start a builder with the `Faer` backend and automatic execution.
    pub fn new() -> Self {
        Self {
            state: SKBuilderState::new(),
            backend: SKBackendKind::Faer,
        }
    }

    /// Select the math backend recorded on reader observability spans.
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

    /// Finalize this builder into an [`SKSafetensorsReader`].
    pub fn build(self) -> Result<SKSafetensorsReader, SKError> {
        <Self as SKBuilder<SKSafetensorsReader>>::build(self)
    }
}

impl Default for SKSafetensorsReaderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SKBuilder<SKSafetensorsReader> for SKSafetensorsReaderBuilder {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKSafetensorsReader, SKError> {
        Ok(SKSafetensorsReader {
            plan: self.state,
            backend: self.backend,
        })
    }
}
