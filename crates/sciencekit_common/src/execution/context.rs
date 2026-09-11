//! The execution context a resolution depends on.

use super::modes::SKAccessPattern;

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
    /// Total element count of the dataset, when known. Used as the in-memory
    /// work-unit size for the parallelism decision.
    pub dataset_elements: Option<u64>,
    /// Bytes occupied by one scalar element (e.g. 8 for `f64`). Used to derive
    /// the resident byte size of a streaming batch for the buffer guard.
    pub scalar_size_bytes: u64,
    /// The documented per-kernel grain — the minimum elements per thread needed
    /// to amortize parallel-dispatch overhead. The caller (an algorithm) injects
    /// the grain of the specific kernel being resolved.
    pub grain: usize,
    /// The access pattern declared by the algorithm.
    pub access_pattern: SKAccessPattern,
    /// An optional batch-size hint (in elements) from the consumer.
    pub batch_size_hint: Option<usize>,
}

impl SKExecutionContext {
    /// Build a context reading the physical environment for memory and cores.
    /// The caller fills the size, pattern and hint fields, which are only known
    /// at operation time. `dataset_elements` is unknown until the caller
    /// supplies it; `scalar_size_bytes` defaults to `f64` (8).
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
            dataset_elements: None,
            scalar_size_bytes: 8,
            // Conservative default grain until the caller injects its kernel's
            // measured value (`sciencekit_math::kernels` documents the per-kernel
            // constants and their calibration provenance, 2026-09-11).
            grain: 1_000_000,
            access_pattern: SKAccessPattern::Sequential,
            batch_size_hint: None,
        }
    }
}