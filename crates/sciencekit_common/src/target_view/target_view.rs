//! The `SKTargetView` enum and its inherent methods.

use ndarray::{ArrayView1, CowArray, Ix1};

use crate::sk_float::SKFloat;

/// A zero-copy view over the targets (labels / responses) of a dataset.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub enum SKTargetView<'a, F: SKFloat> {
    /// Continuous response values in the model scalar.
    Continuous(ArrayView1<'a, F>),
    /// Integer-valued targets.
    Integer(ArrayView1<'a, i64>),
    /// Nominal (textual) symbols, referencing borrowed text.
    Nominal(&'a [&'a str]),
}

impl<'a, F: SKFloat> SKTargetView<'a, F> {
    /// Elevate to continuous values in the model scalar: continuous borrows,
    /// integer is promoted to `F` (exact while the value is representable in
    /// `F`), nominal is rejected (it is categorical, not continuous).
    pub fn as_continuous(&self) -> Result<CowArray<'a, F, Ix1>, crate::SKError> {
        match self {
            SKTargetView::Continuous(view) => Ok(CowArray::from(*view)),
            SKTargetView::Integer(view) => {
                Ok(CowArray::from(view.mapv(|value| {
                    num_traits::cast(value).unwrap_or(F::zero())
                })))
            }
            SKTargetView::Nominal(_) => Err(crate::SKError::UnsupportedRepresentation {
                representation: "nominal",
                suggestion: "encode nominal targets to indices before continuous use",
            }),
        }
    }

    /// The number of targets in the view.
    pub fn len(&self) -> usize {
        match self {
            SKTargetView::Continuous(view) => view.len(),
            SKTargetView::Integer(view) => view.len(),
            SKTargetView::Nominal(text) => text.len(),
        }
    }

    /// Whether the view is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
