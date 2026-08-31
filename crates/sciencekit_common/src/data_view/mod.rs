//! Data boundary: the canonical representation of features (spec
//! `data-view-boundary`).
//!
//! [`SKDataView`] is a `#[non_exhaustive]` enum covering dense and sparse
//! inputs. Public operations accept any type that converts into it via the
//! standard fallible bound (`TryInto`): native types convert infallibly and
//! are promoted automatically by the std blanket; third parties implement
//! [`TryFrom`] for their own types, including fallible conversions whose error
//! flows into the operation's `Result`.

use ndarray::{Array2, ArrayView2};
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

/// Borrowed dense matrix enters without copying (fallible seam, never fails).
impl<'a, F> TryFrom<ArrayView2<'a, F>> for SKDataView<'a, F> {
    type Error = crate::SKError;
    fn try_from(view: ArrayView2<'a, F>) -> Result<Self, Self::Error> {
        Ok(SKDataView::Dense(view))
    }
}

/// Owned dense block enters by borrowing — no data is duplicated.
impl<'a, F> TryFrom<&'a Array2<F>> for SKDataView<'a, F> {
    type Error = crate::SKError;
    fn try_from(arr: &'a Array2<F>) -> Result<Self, Self::Error> {
        Ok(SKDataView::Dense(arr.view()))
    }
}

/// Sparse compressed-row matrix enters through the same boundary.
impl<'a, F> TryFrom<CsMatView<'a, F>> for SKDataView<'a, F> {
    type Error = crate::SKError;
    fn try_from(view: CsMatView<'a, F>) -> Result<Self, Self::Error> {
        Ok(SKDataView::Sparse(view))
    }
}

#[cfg(test)]
mod data_view_tests;
