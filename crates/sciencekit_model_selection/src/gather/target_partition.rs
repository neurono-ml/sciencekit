//! Owned targets produced by row gathering.

use ndarray::Array1;
use sciencekit_common::{SKFloat, SKTargetView};

/// Targets gathered into owned storage, keeping the representation they had.
///
/// Continuous and integer partitions own an array; nominal partitions own the
/// referenced text. A borrowed [`SKTargetView`] can be produced on demand with
/// [`SKTargetPartition::with_view`].
#[derive(Debug, Clone, PartialEq)]
pub enum SKTargetPartition<F: SKFloat> {
    /// Continuous targets in the model scalar.
    Continuous(Array1<F>),
    /// Integer-valued targets.
    Integer(Array1<i64>),
    /// Nominal (textual) targets.
    Nominal(Vec<String>),
}

impl<F: SKFloat> SKTargetPartition<F> {
    /// The number of targets in the partition.
    pub fn number_of_rows(&self) -> usize {
        match self {
            SKTargetPartition::Continuous(values) => values.len(),
            SKTargetPartition::Integer(values) => values.len(),
            SKTargetPartition::Nominal(values) => values.len(),
        }
    }

    /// Borrows the partition as a target view for the duration of `visit`.
    ///
    /// Nominal partitions build a temporary slice of borrowed labels, so the
    /// view never outlives the call.
    pub fn with_view<R>(&self, visit: impl FnOnce(SKTargetView<'_, F>) -> R) -> R {
        match self {
            SKTargetPartition::Continuous(values) => {
                visit(SKTargetView::Continuous(values.view()))
            }
            SKTargetPartition::Integer(values) => visit(SKTargetView::Integer(values.view())),
            SKTargetPartition::Nominal(values) => {
                let references: Vec<&str> = values.iter().map(String::as_str).collect();
                visit(SKTargetView::Nominal(&references))
            }
        }
    }
}
