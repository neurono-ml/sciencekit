//! Structured operation attributes for observability spans (spec `observability`).

use crate::execution::SKExecutionMode;

/// Structured attributes recorded on an operation span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SKOperationAttributes {
    /// The operation name (`fit`, `transform`, `predict`, ...).
    pub operation: &'static str,
    /// The number of rows of the operation input.
    pub rows: usize,
    /// The number of columns of the operation input.
    pub columns: usize,
    /// The execution mode the operation resolved to.
    pub execution_mode: SKExecutionMode,
    /// The math backend the operation uses for heavy dense algebra.
    pub backend: &'static str,
}
