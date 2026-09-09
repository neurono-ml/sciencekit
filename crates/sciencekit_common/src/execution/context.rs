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
