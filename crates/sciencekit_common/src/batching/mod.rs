//! Streaming batches (spec `streaming-batches`).
//!
//! An owned data block ([`SKDataBatch`]) that fully owns its data and can move
//! across threads; a sequential fallible source ([`SKLazySource`]) and an
//! abstract random-access source ([`SKMappableSource`]). Memory-mapped
//! implementations belong to the interop layer, not here.

use ndarray::{Array2, ArrayView1, ArrayView2};

use crate::SKError;

/// An owned streaming block with minimal metadata.
///
/// Owns its data (no borrowing from the source) so it can be moved to another
/// thread while the source advances. Carries its position in the sequence and a
/// final-block indication.
#[derive(Debug, Clone)]
pub struct SKDataBatch<F> {
    data: Array2<F>,
    position: usize,
    is_final: bool,
}

impl<F> SKDataBatch<F> {
    /// Create a batch.
    pub fn new(data: Array2<F>, position: usize, is_final: bool) -> Self {
        SKDataBatch {
            data,
            position,
            is_final,
        }
    }

    /// Borrow the batch's data.
    pub fn data(&self) -> ArrayView2<F> {
        self.data.view()
    }

    /// The zero-based position of this batch in the sequence.
    pub fn position(&self) -> usize {
        self.position
    }

    /// Whether this is the final block of a finite source.
    pub fn is_final(&self) -> bool {
        self.is_final
    }
}

/// A sequential streaming source exposed as fallible block iteration.
///
/// Intermediate read failures yield an error from the central taxonomy instead
/// of panicking; the consumer decides whether to stop or handle it.
pub trait SKLazySource<F> {
    /// The iteration error type; converts from the central [`SKError`].
    type Error: From<SKError>;
    /// Produce the next batch, `None` at the end of a finite source.
    fn next_batch(&mut self) -> Result<Option<SKDataBatch<F>>, Self::Error>;
}

/// An abstract random-access source: direct positional access to data units,
/// independent of the storage mechanism (memmap arrives at interop).
pub trait SKMappableSource<F> {
    /// The access error type; converts from the central [`SKError`].
    type Error: From<SKError>;
    /// The number of data rows (units).
    fn number_of_rows(&self) -> usize;
    /// Access a row by index without traversing previous rows.
    fn row(&self, index: usize) -> Result<ArrayView1<'_, F>, Self::Error>;
}

#[cfg(test)]
mod batching_tests;
