//! The `SKTargetView` enum and its inherent methods.

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
