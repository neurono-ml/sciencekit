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
    /// The prefetch buffer depth for streaming — the number of batches read
    /// ahead of compute. Double buffering is `1` (one batch computing, one
    /// prefetched); depth greater than one is future work.
    pub buffer_depth: usize,
}
