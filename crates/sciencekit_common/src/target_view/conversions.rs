//! Conversions into [`SKTargetView`] from native target representations.

use ndarray::ArrayView1;

use super::SKTargetView;
use crate::sk_float::SKFloat;

/// Integer targets may be provided directly (elevated on demand).
impl<'a, F: SKFloat> TryFrom<ArrayView1<'a, i64>> for SKTargetView<'a, F> {
    type Error = crate::SKError;
    fn try_from(view: ArrayView1<'a, i64>) -> Result<Self, Self::Error> {
        Ok(SKTargetView::Integer(view))
    }
}

/// Continuous targets may be provided directly in the model scalar.
impl<'a, F: SKFloat> TryFrom<ArrayView1<'a, F>> for SKTargetView<'a, F> {
    type Error = crate::SKError;
    fn try_from(view: ArrayView1<'a, F>) -> Result<Self, Self::Error> {
        Ok(SKTargetView::Continuous(view))
    }
}

/// Nominal (textual) targets reference borrowed text.
impl<'a, F: SKFloat> TryFrom<&'a [&'a str]> for SKTargetView<'a, F> {
    type Error = crate::SKError;
    fn try_from(labels: &'a [&'a str]) -> Result<Self, Self::Error> {
        Ok(SKTargetView::Nominal(labels))
    }
}
