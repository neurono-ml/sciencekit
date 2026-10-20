//! Conversions into [`SKCategoricalView`] from native representations.
//!
//! The seam mirrors the numeric boundary: native types convert through the
//! standard fallible bound with a [`crate::SKError`] error, and third parties
//! can plug their own (including fallible) conversions. The
//! `From<SKNominalMatrixView>` impl lives on the enum definition.

use ndarray::{Array2, ArrayView2};

use super::{SKCategoricalView, SKNominalMatrixView};

/// A borrowed row-major matrix of labels enters without copying.
impl<'a> TryFrom<&'a [&'a [&'a str]]> for SKCategoricalView<'a> {
    type Error = crate::SKError;
    fn try_from(rows: &'a [&'a [&'a str]]) -> Result<Self, Self::Error> {
        Ok(SKCategoricalView::Nominal(SKNominalMatrixView::new(rows)))
    }
}

/// A borrowed integer-coded matrix enters through the same boundary.
impl<'a> TryFrom<ArrayView2<'a, i64>> for SKCategoricalView<'a> {
    type Error = crate::SKError;
    fn try_from(view: ArrayView2<'a, i64>) -> Result<Self, Self::Error> {
        Ok(SKCategoricalView::Integer(view))
    }
}

/// An owned integer-coded block enters by borrowing — no data is duplicated.
impl<'a> TryFrom<&'a Array2<i64>> for SKCategoricalView<'a> {
    type Error = crate::SKError;
    fn try_from(array: &'a Array2<i64>) -> Result<Self, Self::Error> {
        Ok(SKCategoricalView::Integer(array.view()))
    }
}
