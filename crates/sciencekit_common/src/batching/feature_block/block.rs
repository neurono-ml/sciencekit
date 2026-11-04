//! The owned twin of [`SKDataView`]: a dense matrix or a CSR matrix.

use std::mem::size_of;

use ndarray::Array2;
use sprs::CsMat;

use crate::SKDataView;

/// An owned streaming feature block with one of two representations.
///
/// Mirrors [`SKDataView`]'s variants so representation dispatch reads the same
/// in the in-memory and streaming regimes. The block fully owns its data, so it
/// can be moved to another thread while the source advances. `#[non_exhaustive]`
/// leaves room for future representations.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum SKFeatureBlock<F> {
    /// Dense, row-major feature matrix.
    Dense(Array2<F>),
    /// Sparse compressed-row feature matrix.
    ///
    /// The block's accessors assume row-compressed storage; construct blocks
    /// through the batch constructors, which assert the CSR layout, rather
    /// than building a CSC matrix directly.
    Sparse(CsMat<F>),
}

impl<F> SKFeatureBlock<F> {
    /// Borrow the block as a [`SKDataView`], without copying or converting.
    pub fn view(&self) -> SKDataView<'_, F> {
        match self {
            SKFeatureBlock::Dense(matrix) => SKDataView::Dense(matrix.view()),
            SKFeatureBlock::Sparse(matrix) => SKDataView::Sparse(matrix.view()),
        }
    }

    /// The number of rows, whichever representation the block carries.
    pub fn number_of_rows(&self) -> usize {
        match self {
            SKFeatureBlock::Dense(matrix) => matrix.nrows(),
            SKFeatureBlock::Sparse(matrix) => matrix.rows(),
        }
    }

    /// The number of columns, whichever representation the block carries.
    pub fn number_of_columns(&self) -> usize {
        match self {
            SKFeatureBlock::Dense(matrix) => matrix.ncols(),
            SKFeatureBlock::Sparse(matrix) => matrix.cols(),
        }
    }

    /// Whether the block carries a CSR matrix.
    pub fn is_sparse(&self) -> bool {
        matches!(self, SKFeatureBlock::Sparse(_))
    }

    /// The resident byte size of the stored buffers, a lower bound.
    ///
    /// Dense blocks count the logical `rows × columns × size_of(scalar)`; sparse
    /// blocks sum the stored element, index and pointer array lengths. The
    /// figure excludes any spare capacity beyond those lengths, so it is a
    /// lower bound rather than an exact allocator measurement.
    pub fn byte_size(&self) -> usize {
        match self {
            SKFeatureBlock::Dense(matrix) => matrix.len().saturating_mul(size_of::<F>()),
            SKFeatureBlock::Sparse(matrix) => matrix
                .data()
                .len()
                .saturating_mul(size_of::<F>())
                .saturating_add(matrix.indices().len().saturating_mul(size_of::<usize>()))
                .saturating_add(
                    matrix
                        .indptr()
                        .raw_storage()
                        .len()
                        .saturating_mul(size_of::<usize>()),
                ),
        }
    }
}
