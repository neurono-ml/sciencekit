//! The streaming driver's mutable state (spec `ridge-regression`).

use sciencekit_common::{SKError, SKFloat};

use crate::ridge::gram::SKCenteredGramState;

/// The state carried through the supervised prefetch pipeline: the running
/// centered Gram, a parked error and the machine facts the per-batch
/// parallelism needs.
pub(super) struct StreamingCenteredGramState<F> {
    /// The feature column count, `0` until the first batch is seen.
    pub(super) columns_count: usize,
    /// The accumulated centered cross products.
    pub(super) gram: SKCenteredGramState<F>,
    /// A shape violation parked by the infallible driver closure.
    pub(super) error: Option<SKError>,
    /// Available memory, for the per-batch parallelism decision.
    pub(super) available_memory_bytes: u64,
    /// Core count, for the per-batch parallelism decision.
    pub(super) cpu_cores: usize,
}

impl<F: SKFloat> StreamingCenteredGramState<F> {
    /// A fresh state before any batch is seen.
    pub(super) fn new(available_memory_bytes: u64, cpu_cores: usize) -> Self {
        Self {
            columns_count: 0,
            gram: SKCenteredGramState::empty(0),
            error: None,
            available_memory_bytes,
            cpu_cores,
        }
    }
}
