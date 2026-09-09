//! Conversions into [`SKTargetView`] from native target representations.

use ndarray::ArrayView1;

use super::SKTargetView;

/// Integer targets may be provided directly (elevated on demand).
impl<'a> TryFrom<ArrayView1<'a, i64>> for SKTargetView<'a> {
    type Error = crate::SKError;
    fn try_from(view: ArrayView1<'a, i64>) -> Result<Self, Self::Error> {
        Ok(SKTargetView::Integer(view))
    }
}

/// Continuous targets may be provided directly.
impl<'a> TryFrom<ArrayView1<'a, f64>> for SKTargetView<'a> {
    type Error = crate::SKError;
    fn try_from(view: ArrayView1<'a, f64>) -> Result<Self, Self::Error> {
        Ok(SKTargetView::Continuous(view))
    }
}

/// Nominal (textual) targets reference borrowed text.
impl<'a> TryFrom<&'a [&'a str]> for SKTargetView<'a> {
    type Error = crate::SKError;
    fn try_from(labels: &'a [&'a str]) -> Result<Self, Self::Error> {
        Ok(SKTargetView::Nominal(labels))
    }
}
