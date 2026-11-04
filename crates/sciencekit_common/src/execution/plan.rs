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
    /// The memory available to the operation, in bytes, copied from the
    /// [`SKExecutionContext`](super::SKExecutionContext) at resolution. The
    /// provider entry points use it to reject a batch whose doubled resident
    /// size would not fit before the update closure sees it.
    pub available_memory_bytes: u64,
}
