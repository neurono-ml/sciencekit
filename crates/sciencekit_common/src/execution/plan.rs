//! The resolved execution plan produced by resolution.

use super::modes::SKExecutionMode;

/// The resolved plan for one operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SKExecutionPlan {
    /// The chosen execution mode.
    pub mode: SKExecutionMode,
    /// The number of compute threads to use.
    pub parallelism: usize,
    /// The batch size for streaming modes, when applicable.
    pub batch_size: Option<usize>,
}
