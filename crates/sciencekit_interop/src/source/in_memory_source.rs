//! An in-memory reopenable tabular source (spec `dataset-interop-core`).

use sciencekit_common::SKError;

use super::SKTabularSource;
use crate::batch::SKTabularBatch;

/// A reopenable source over batches already held in memory.
///
/// Useful for tests, examples and small datasets: every [`open`](SKTabularSource::open)
/// resets the cursor, and each batch is cloned out so the caller owns it.
#[derive(Debug, Clone)]
pub struct SKInMemoryTabularSource {
    batches: Vec<SKTabularBatch>,
    cursor: usize,
}

impl SKInMemoryTabularSource {
    /// Build a source from owned batches.
    pub fn new(batches: Vec<SKTabularBatch>) -> Self {
        SKInMemoryTabularSource { batches, cursor: 0 }
    }

    /// The number of batches the source holds.
    pub fn len(&self) -> usize {
        self.batches.len()
    }

    /// Whether the source holds no batch.
    pub fn is_empty(&self) -> bool {
        self.batches.is_empty()
    }
}

impl SKTabularSource for SKInMemoryTabularSource {
    const IS_REOPENABLE: bool = true;

    fn open(&mut self) -> Result<(), SKError> {
        self.cursor = 0;
        Ok(())
    }

    fn next_batch(&mut self) -> Result<Option<SKTabularBatch>, SKError> {
        let batch = self.batches.get(self.cursor).cloned();
        if batch.is_some() {
            self.cursor += 1;
        }
        Ok(batch)
    }
}
