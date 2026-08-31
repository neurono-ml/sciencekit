//! Target boundary: continuous/integer/nominal target representation
//! (spec `data-view-boundary`).
//!
//! Storage ≠ interpretation: `[1,2,3]` is continuous for a regressor and
//! categorical for a classifier. The view describes *how* data is stored.
//! Continuous targets use `f64` independently of the feature dtype `F`
//! (design decision 4). Integer targets elevate losslessly to continuous;
//! nominal targets borrow their text.

use ndarray::{ArrayView1, CowArray, Ix1};

/// A zero-copy view over the targets (labels / responses) of a dataset.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum SKTargetView<'a> {
    /// Continuous response values (independent of the feature dtype).
    Continuous(ArrayView1<'a, f64>),
    /// Integer-valued targets.
    Integer(ArrayView1<'a, i64>),
    /// Nominal (textual) symbols, referencing borrowed text.
    Nominal(&'a [&'a str]),
}

impl<'a> SKTargetView<'a> {
    /// Elevate to continuous values losslessly: continuous borrows, integer is
    /// promoted to `f64` (exact for the `i64` range representable in f64),
    /// nominal is rejected (it is categorical, not continuous).
    pub fn as_continuous(&self) -> Result<CowArray<'a, f64, Ix1>, crate::SKError> {
        match self {
            SKTargetView::Continuous(view) => Ok(CowArray::from(*view)),
            SKTargetView::Integer(view) => Ok(CowArray::from(view.mapv(|v| v as f64))),
            SKTargetView::Nominal(_) => Err(crate::SKError::UnsupportedRepresentation {
                representation: "nominal",
                suggestion: "encode nominal targets to indices before continuous use",
            }),
        }
    }

    /// The number of targets in the view.
    pub fn len(&self) -> usize {
        match self {
            SKTargetView::Continuous(v) => v.len(),
            SKTargetView::Integer(v) => v.len(),
            SKTargetView::Nominal(v) => v.len(),
        }
    }

    /// Whether the view is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

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

#[cfg(test)]
mod target_view_tests;
