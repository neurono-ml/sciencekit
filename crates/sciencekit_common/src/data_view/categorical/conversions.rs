//! Conversions into [`SKCategoricalView`] from native representations.
//!
//! The seam mirrors the numeric boundary: native types convert through the
//! standard fallible bound and third parties can plug their own (including
//! fallible) conversions. The `From<ArrayView2<i64>>` and
//! `From<SKNominalMatrixView>` impls live on the enum definition in
//! `categorical_view.rs`, so only the borrowed owned-block conversion is added
//! here.

use ndarray::Array2;

use super::{SKCategoricalView, SKNominalMatrixView};

/// A borrowed row-major matrix of labels enters without copying.
impl<'a> TryFrom<&'a [&'a [&'a str]]> for SKCategoricalView<'a> {
    type Error = crate::SKError;
    fn try_from(rows: &'a [&'a [&'a str]]) -> Result<Self, Self::Error> {
        Ok(SKCategoricalView::Nominal(SKNominalMatrixView::new(rows)))
    }
}

/// An owned integer-coded block enters by borrowing — no data is duplicated.
impl<'a> From<&'a Array2<i64>> for SKCategoricalView<'a> {
    fn from(array: &'a Array2<i64>) -> Self {
        SKCategoricalView::Integer(array.view())
    }
}
