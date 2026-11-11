//! The resolved execution plan produced by resolution.

use crate::device::SKFallbackReason;
use crate::observability::SKBackendKind;

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
    /// size would not fit before the update closure sees it — the `2` follows
    /// from the double-buffer contract with [`buffer_depth`](SKExecutionPlan::buffer_depth)
    /// pinned to `1`, and must become `(depth + 1) ×` if the depth is raised.
    pub available_memory_bytes: u64,
    /// The backend the plan runs on — the selected device kind, or the host
    /// backend when no device serves the operation.
    pub backend: SKBackendKind,
    /// Why the plan did not select a device, when one was requested but is not
    /// usable. `None` means "no device was requested, or the selected device is
    /// the configured one".
    pub fallback_reason: Option<SKFallbackReason>,
}
