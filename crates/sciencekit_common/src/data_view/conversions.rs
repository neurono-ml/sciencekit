//! Conversions into [`SKDataView`] from native representations.

use ndarray::{Array2, ArrayView2};
use sprs::CsMatView;

use super::SKDataView;

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
