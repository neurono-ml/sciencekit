//! The owned supervised streaming block (spec `streaming-batches`).

use ndarray::{Array1, ArrayView1};
use sprs::CsMat;

use super::SKFeatureBlock;
use crate::{SKDataView, SKError};

/// An owned supervised streaming block: a feature block paired with its
/// targets.
///
/// Like [`SKDataBatch`](super::SKDataBatch) it fully owns its data (no borrowing
/// from the source) so the pair can be moved to another thread while the source
/// advances; it carries its position in the sequence and a final-block
/// indication. Construction rejects a target count different from the block's
/// row count.
#[derive(Debug, Clone)]
pub struct SKSupervisedBatch<F> {
    block: SKFeatureBlock<F>,
    targets: Array1<F>,
    position: usize,
    is_final: bool,
}

impl<F> SKSupervisedBatch<F> {
    /// Create a supervised batch from a dense matrix — the dense alias of
    /// [`Self::new_dense`].
    pub fn new(
        features: ndarray::Array2<F>,
        targets: Array1<F>,
        position: usize,
        is_final: bool,
    ) -> Result<Self, SKError> {
        Self::new_dense(features, targets, position, is_final)
    }

    /// Create a supervised batch that owns a dense matrix.
    pub fn new_dense(
        features: ndarray::Array2<F>,
        targets: Array1<F>,
        position: usize,
        is_final: bool,
    ) -> Result<Self, SKError> {
        Self::from_block(SKFeatureBlock::Dense(features), targets, position, is_final)
    }

    /// Create a supervised batch that owns a CSR matrix.
    pub fn new_sparse(
        features: CsMat<F>,
        targets: Array1<F>,
        position: usize,
        is_final: bool,
    ) -> Result<Self, SKError> {
        debug_assert!(
            features.is_csr(),
            "a sparse supervised batch must store its matrix in CSR"
        );
        Self::from_block(
            SKFeatureBlock::Sparse(features),
            targets,
            position,
            is_final,
        )
    }

    fn from_block(
        block: SKFeatureBlock<F>,
        targets: Array1<F>,
        position: usize,
        is_final: bool,
    ) -> Result<Self, SKError> {
        let rows = block.number_of_rows();
        if targets.len() != rows {
            return Err(SKError::ShapeMismatch {
                expected: vec![rows],
                found: vec![targets.len()],
            });
        }
        Ok(SKSupervisedBatch {
            block,
            targets,
            position,
            is_final,
        })
    }

    /// Borrow the batch's feature block as a [`SKDataView`].
    pub fn view(&self) -> SKDataView<'_, F> {
        self.block.view()
    }

    /// The number of rows in the feature block.
    pub fn number_of_rows(&self) -> usize {
        self.block.number_of_rows()
    }

    /// The number of columns in the feature block.
    pub fn number_of_columns(&self) -> usize {
        self.block.number_of_columns()
    }

    /// Whether the feature block carries a CSR matrix.
    pub fn is_sparse(&self) -> bool {
        self.block.is_sparse()
    }

    /// The resident byte size of the feature block's stored buffers, a lower
    /// bound; targets are not included.
    pub fn byte_size(&self) -> usize {
        self.block.byte_size()
    }

    /// Borrow the batch's targets.
    pub fn targets(&self) -> ArrayView1<'_, F> {
        self.targets.view()
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
