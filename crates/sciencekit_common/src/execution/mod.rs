//! Execution planning (spec `execution-planning`).
//!
//! Separates the consumer-declared **intent** ([`SKExecutionMode`]) from the
//! effective **plan** ([`SKExecutionPlan`]) resolved per operation. Resolution
//! is a pure function over an injectable [`SKExecutionContext`] — deterministic
//! and testable without a real machine. An explicitly requested mode
//! incompatible with the algorithm's declared access pattern fails with a
//! specific error; automatic intent never does.

use crate::SKError;

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

/// The explicit context a resolution depends on. The default constructor reads
/// the physical machine (via `sysinfo`); tests inject simulated values.
#[derive(Debug, Clone)]
pub struct SKExecutionContext {
    /// Free memory in bytes.
    pub available_memory_bytes: u64,
    /// Number of CPU cores.
    pub cpu_cores: usize,
    /// Size of the dataset for the current operation, in bytes.
    pub dataset_size_bytes: u64,
    /// The access pattern declared by the algorithm.
    pub access_pattern: SKAccessPattern,
    /// An optional batch-size hint from the consumer.
    pub batch_size_hint: Option<usize>,
}

impl SKExecutionContext {
    /// Build a context reading the physical environment for memory and cores.
    /// The caller fills `dataset_size_bytes`, `access_pattern` and
    /// `batch_size_hint`, which are only known at operation time.
    pub fn real() -> Self {
        let mut sys = sysinfo::System::new_all();
        sys.refresh_memory();
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        SKExecutionContext {
            available_memory_bytes: sys.available_memory(),
            cpu_cores: cores,
            dataset_size_bytes: 0,
            access_pattern: SKAccessPattern::Sequential,
            batch_size_hint: None,
        }
    }
}

/// Resolve an execution plan from an intent and a context.
///
/// Pure and deterministic: the same `(intent, context)` always yields the same
/// plan. Automatic intent never produces an incompatibility error; an explicit
/// mode incompatible with the declared access pattern does.
pub fn sk_resolve_execution_plan(
    intent: SKExecutionMode,
    context: &SKExecutionContext,
) -> Result<SKExecutionPlan, SKError> {
    let parallelism = context.cpu_cores.max(1);
    match intent {
        SKExecutionMode::Automatic => {
            let fits_in_memory = context.dataset_size_bytes <= context.available_memory_bytes;
            let mode = if fits_in_memory {
                SKExecutionMode::InProcessSynchronous
            } else {
                match context.access_pattern {
                    SKAccessPattern::RandomAccess => SKExecutionMode::OutOfCoreMemoryMapped,
                    SKAccessPattern::Sequential | SKAccessPattern::Iterative => {
                        SKExecutionMode::OutOfCoreStreaming
                    }
                }
            };
            let batch_size = if mode == SKExecutionMode::OutOfCoreStreaming {
                context.batch_size_hint
            } else {
                None
            };
            Ok(SKExecutionPlan {
                mode,
                parallelism,
                batch_size,
            })
        }
        SKExecutionMode::InProcessSynchronous
        | SKExecutionMode::InProcessAsynchronous
        | SKExecutionMode::OutOfCoreMemoryMapped => Ok(SKExecutionPlan {
            mode: intent,
            parallelism,
            batch_size: None,
        }),
        SKExecutionMode::OutOfCoreStreaming => {
            if context.access_pattern == SKAccessPattern::RandomAccess {
                Err(SKError::ExecutionModeIncompatible {
                    mode: "out-of-core-streaming",
                    pattern: "random-access",
                })
            } else {
                Ok(SKExecutionPlan {
                    mode: intent,
                    parallelism,
                    batch_size: context.batch_size_hint,
                })
            }
        }
    }
}

#[cfg(test)]
mod execution_tests;
