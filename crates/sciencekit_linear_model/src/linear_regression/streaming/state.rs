//! The streaming driver's mutable state (spec `linear-regression`).

use sciencekit_common::SKError;
use sciencekit_common::SKFloat;

use crate::linear_regression::gram::SKGramState;

/// The state carried through the supervised prefetch pipeline: the running Gram
/// state, a parked error and the machine facts the per-batch parallelism needs.
pub(super) struct StreamingGramState<F> {
    /// The feature column count, `0` until the first batch is seen.
    pub(super) columns_count: usize,
    /// The accumulated cross products.
    pub(super) gram: SKGramState<F>,
    /// A shape violation parked by the infallible driver closure.
    pub(super) error: Option<SKError>,
    /// Available memory, for the per-batch parallelism decision.
    pub(super) available_memory_bytes: u64,
    /// Core count, for the per-batch parallelism decision.
    pub(super) cpu_cores: usize,
}

impl<F: SKFloat> StreamingGramState<F> {
    /// A fresh state before any batch is seen.
    pub(super) fn new(available_memory_bytes: u64, cpu_cores: usize) -> Self {
        Self {
            columns_count: 0,
            gram: SKGramState::empty(0),
            error: None,
            available_memory_bytes,
            cpu_cores,
        }
    }
}
