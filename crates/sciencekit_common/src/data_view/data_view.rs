//! The `SKDataView` enum and its inherent methods.

use ndarray::ArrayView2;
use sprs::CsMatView;

/// A zero-copy view over the features of a dataset.
///
/// Native conversions borrow the underlying data — no copy of the elements is
/// performed. Representation dispatch happens exactly once per operation via
/// [`SKDataView::representation`] / [`SKDataView::as_dense`].
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum SKDataView<'a, F> {
    /// Dense, contiguous (row-major) feature matrix.
    Dense(ArrayView2<'a, F>),
    /// Sparse compressed-row feature matrix.
    Sparse(CsMatView<'a, F>),
}

impl<'a, F> SKDataView<'a, F> {
    /// The representation label, used for one-time dispatch and diagnostics.
    pub fn representation(&self) -> &'static str {
        match self {
            SKDataView::Dense(_) => "dense",
            SKDataView::Sparse(_) => "csr",
        }
    }

    /// Dense-only consumers dispatch here exactly once: returns the dense view
    /// or rejects a sparse input with a precise unsupported-representation
    /// error *before* processing any element.
    pub fn as_dense(&self) -> Result<ArrayView2<'a, F>, crate::SKError> {
        match self {
            SKDataView::Dense(view) => Ok(*view),
            SKDataView::Sparse(_) => Err(crate::SKError::UnsupportedRepresentation {
                representation: "csr",
                suggestion: "convert the sparse matrix to dense with to_dense",
            }),
        }
    }
}
