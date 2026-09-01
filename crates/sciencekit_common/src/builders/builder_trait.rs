//! The mandatory builder contract (spec `base-builders`, PRD §2.1, §5.3).

use crate::SKError;
use crate::execution::SKExecutionMode;

/// The mandatory builder contract for estimators, transformers and algorithms.
///
/// The execution intent defaults to [`SKExecutionMode::Automatic`] through
/// [`super::SKBuilderState`]; `build` validates the accumulated configuration and
/// returns the constructed model through the central error taxonomy, never
/// panicking.
pub trait SKBuilder<Model> {
    /// Set the execution intent; the default is `Automatic`.
    fn execution_mode(&mut self, mode: SKExecutionMode) -> &mut Self;

    /// Construct the model, validating the accumulated configuration.
    fn build(self) -> Result<Model, SKError>;
}
