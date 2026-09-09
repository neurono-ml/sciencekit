//! The owned streaming data block.

use ndarray::{Array2, ArrayView2};

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
