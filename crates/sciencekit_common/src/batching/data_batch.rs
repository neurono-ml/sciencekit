//! The owned streaming data block.

use ndarray::Array2;
use sprs::CsMat;

use super::SKFeatureBlock;
use crate::SKDataView;

/// An owned streaming block with minimal metadata.
///
/// Owns a [`SKFeatureBlock`] (no borrowing from the source) so it can be moved
/// to another thread while the source advances. Carries its position in the
/// sequence and a final-block indication.
#[derive(Debug, Clone)]
pub struct SKDataBatch<F> {
    block: SKFeatureBlock<F>,
    position: usize,
    is_final: bool,
}

impl<F> SKDataBatch<F> {
    /// Create a batch from a dense matrix — the dense alias of [`Self::new_dense`].
    pub fn new(data: Array2<F>, position: usize, is_final: bool) -> Self {
        Self::new_dense(data, position, is_final)
    }

    /// Create a batch that owns a dense matrix.
    pub fn new_dense(data: Array2<F>, position: usize, is_final: bool) -> Self {
        SKDataBatch {
            block: SKFeatureBlock::Dense(data),
            position,
            is_final,
        }
    }

    /// Create a batch that owns a CSR matrix.
    pub fn new_sparse(data: CsMat<F>, position: usize, is_final: bool) -> Self {
        SKDataBatch {
            block: SKFeatureBlock::Sparse(data),
            position,
            is_final,
        }
    }

    /// Borrow the batch's feature block as a [`SKDataView`].
    pub fn view(&self) -> SKDataView<'_, F> {
        self.block.view()
    }

    /// The number of rows in the block.
    pub fn number_of_rows(&self) -> usize {
        self.block.number_of_rows()
    }

    /// The number of columns in the block.
    pub fn number_of_columns(&self) -> usize {
        self.block.number_of_columns()
    }

    /// Whether the block carries a CSR matrix.
    pub fn is_sparse(&self) -> bool {
        self.block.is_sparse()
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
