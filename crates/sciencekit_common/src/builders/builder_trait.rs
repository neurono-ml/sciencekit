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
    ///
    /// Accepts any value convertible to [`SKExecutionMode`] via `TryInto`, so a
    /// caller passes the enum directly and a mis-typed string cannot compile.
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>;

    /// Construct the model, validating the accumulated configuration.
    fn build(self) -> Result<Model, SKError>;
}
