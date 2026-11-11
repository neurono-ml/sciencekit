//! Builder for [`super::SKJsonDebugWriter`] (spec `json-debug-format`).

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode,
    builders::{SKBuilder, SKBuilderState},
};

use super::{SKJsonDebugWriter, SKJsonFormatMode};

/// Builder configuring a [`SKJsonDebugWriter`].
///
/// Defaults to [`SKJsonFormatMode::Pretty`], [`SKBackendKind::Faer`] and
/// [`SKExecutionMode::Automatic`].
#[derive(Debug, Clone, Copy)]
pub struct SKJsonDebugWriterBuilder {
    pub(super) state: SKBuilderState,
    pub(super) format_mode: SKJsonFormatMode,
    pub(super) backend: SKBackendKind,
}

impl SKJsonDebugWriterBuilder {
    /// Start a builder with pretty formatting, the `Faer` backend and automatic execution.
    pub fn new() -> Self {
        Self {
            state: SKBuilderState::new(),
            format_mode: SKJsonFormatMode::Pretty,
            backend: SKBackendKind::Faer,
        }
    }

    /// Select pretty (multi-line) or compact (single-line) output formatting.
    pub fn format_mode(mut self, format_mode: SKJsonFormatMode) -> Self {
        self.format_mode = format_mode;
        self
    }

    /// Select the math backend recorded on writer spans.
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

impl Default for SKJsonDebugWriterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl SKBuilder<SKJsonDebugWriter> for SKJsonDebugWriterBuilder {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKJsonDebugWriter, SKError> {
        Ok(SKJsonDebugWriter {
            plan: self.state,
            format_mode: self.format_mode,
            backend: self.backend,
        })
    }
}
