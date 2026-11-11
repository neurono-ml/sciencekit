//! Builder for [`super::SKJsonDebugReader`] (spec `json-debug-format`).

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode,
    builders::{SKBuilder, SKBuilderState},
};

use super::SKJsonDebugReader;

/// Builder configuring a [`SKJsonDebugReader`].
///
/// Defaults to [`SKBackendKind::Faer`] and [`SKExecutionMode::Automatic`].
#[derive(Debug, Clone, Copy)]
pub struct SKJsonDebugReaderBuilder {
    pub(super) state: SKBuilderState,
    pub(super) backend: SKBackendKind,
}

impl SKJsonDebugReaderBuilder {
    /// Start a builder with the `Faer` backend and automatic execution.
    pub fn new() -> Self {
        Self {
            state: SKBuilderState::new(),
            backend: SKBackendKind::Faer,
        }
    }

    /// Select the math backend recorded on reader spans.
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

impl Default for SKJsonDebugReaderBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SKBuilder<SKJsonDebugReader> for SKJsonDebugReaderBuilder {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKJsonDebugReader, SKError> {
        Ok(SKJsonDebugReader {
            plan: self.state,
            backend: self.backend,
        })
    }
}
