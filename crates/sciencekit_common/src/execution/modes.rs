//! The access-pattern and execution-mode intents.

/// The access pattern an algorithm declares for its data (PRD §4.4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKAccessPattern {
    /// Sequential scan in batches.
    Sequential,
    /// O(1) random access by position.
    RandomAccess,
    /// Iterative passes over the data.
    Iterative,
}

/// The execution intent declared by the consumer (PRD §5.3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum SKExecutionMode {
    /// The library decides a compatible mode automatically (default).
    Automatic,
    /// Dataset fits in memory; eager, in-process, synchronous.
    InProcessSynchronous,
    /// Async I/O source; Tokio orchestrates, compute runs on a CPU pool.
    InProcessAsynchronous,
    /// Dataset exceeds memory; sequential streaming in batches.
    OutOfCoreStreaming,
    /// Dataset exceeds memory; random access via memory mapping.
    OutOfCoreMemoryMapped,
}
