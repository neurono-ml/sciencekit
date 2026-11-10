//! Conversions between owned blocks, native matrices and borrowed views.
//!
//! The conversions into a block are the seam the pipeline stage adapter uses:
//! every existing transformer output (a dense `Array2` or a CSR `CsMat`) has a
//! lossless, representation-preserving path into [`SKFeatureBlock`]. The
//! conversion out of a block borrows, so a stage never copies its input.

use ndarray::Array2;
use sprs::CsMat;

use super::SKFeatureBlock;
use crate::data_view::SKDataView;

/// A dense matrix becomes a dense block, keeping its row-major layout.
impl<F> From<Array2<F>> for SKFeatureBlock<F> {
    fn from(matrix: Array2<F>) -> Self {
        SKFeatureBlock::Dense(matrix)
    }
}

/// A CSR matrix becomes a sparse block without changing representation.
impl<F> From<CsMat<F>> for SKFeatureBlock<F> {
    fn from(matrix: CsMat<F>) -> Self {
        SKFeatureBlock::Sparse(matrix)
    }
}

/// A borrowed view becomes an owned block, preserving its representation.
///
/// This is the single copy at the pipeline boundary: the caller keeps a
/// zero-copy view of its data, and the block owns what later stages produce.
/// A sparse view stays sparse — the conversion never densifies.
impl<'a, F: Clone> From<SKDataView<'a, F>> for SKFeatureBlock<F> {
    fn from(view: SKDataView<'a, F>) -> Self {
        match view {
            SKDataView::Dense(matrix) => SKFeatureBlock::Dense(matrix.to_owned()),
            SKDataView::Sparse(matrix) => SKFeatureBlock::Sparse(matrix.to_owned()),
        }
    }
}

/// A block enters the shared operation seam by borrowing — no copy is made.
impl<'a, F> TryFrom<&'a SKFeatureBlock<F>> for SKDataView<'a, F> {
    type Error = crate::SKError;

    fn try_from(block: &'a SKFeatureBlock<F>) -> Result<Self, Self::Error> {
        Ok(block.view())
    }
}

/// Conversion of a transformer output into the owned pipeline medium.
///
/// A dedicated trait rather than `Into<SKFeatureBlock<F>>` because the orphan
/// rule forbids a crate from implementing `From` for two foreign types: the
/// impute crate owns its output enums but not [`SKFeatureBlock`]. Implementing
/// this trait for a local output type is always allowed, and the pipeline stage
/// adapter bounds its output on it.
pub trait SKIntoFeatureBlock<F> {
    /// Consume the output and wrap it as a pipeline block, preserving the
    /// representation it already carries.
    fn into_feature_block(self) -> SKFeatureBlock<F>;
}

impl<F> SKIntoFeatureBlock<F> for Array2<F> {
    fn into_feature_block(self) -> SKFeatureBlock<F> {
        SKFeatureBlock::Dense(self)
    }
}

impl<F> SKIntoFeatureBlock<F> for CsMat<F> {
    fn into_feature_block(self) -> SKFeatureBlock<F> {
        SKFeatureBlock::Sparse(self)
    }
}
