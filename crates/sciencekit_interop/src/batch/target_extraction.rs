//! Extraction of the selected target column through the library's target view
//! (spec `dataset-interop-core`).

use ndarray::{Array1, ArrayView1};
use num_traits::cast;
use sciencekit_common::{SKError, SKFloat, SKLabelTable, SKTargetView};

/// An owned target column, ready to be exposed through [`SKTargetView`].
///
/// Continuous and integer targets are stored as scalar vectors; a nominal
/// target keeps both its deterministic label table and the canonical indices,
/// so a classifier can round-trip labels without ambiguity.
#[derive(Debug, Clone, PartialEq)]
pub enum SKTargetExtraction<F: SKFloat> {
    /// Continuous response values in the model scalar.
    Continuous(Array1<F>),
    /// Integer-valued targets, including category codes.
    Integer(Array1<i64>),
    /// Nominal labels with their canonical index mapping.
    Nominal {
        /// Canonical zero-based class indices; nulls carry the documented
        /// missing code when the policy allows them.
        indices: Array1<i64>,
        /// The original labels per row; `None` marks a missing entry.
        labels: Vec<Option<String>>,
        /// The deterministic label-to-index table.
        table: SKLabelTable,
    },
}

impl<F: SKFloat> SKTargetExtraction<F> {
    /// The number of targets.
    pub fn len(&self) -> usize {
        match self {
            SKTargetExtraction::Continuous(values) => values.len(),
            SKTargetExtraction::Integer(values) => values.len(),
            SKTargetExtraction::Nominal { indices, .. } => indices.len(),
        }
    }

    /// Whether the extraction is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The deterministic label table, when the target is nominal.
    pub fn label_table(&self) -> Option<&SKLabelTable> {
        match self {
            SKTargetExtraction::Nominal { table, .. } => Some(table),
            _ => None,
        }
    }

    /// The canonical indices, when the target is nominal.
    pub fn canonical_indices(&self) -> Option<ArrayView1<'_, i64>> {
        match self {
            SKTargetExtraction::Nominal { indices, .. } => Some(indices.view()),
            _ => None,
        }
    }

    /// Run a consumer with the borrowed [`SKTargetView`].
    ///
    /// A nominal target with missing entries cannot be represented as borrowed
    /// text, so it is exposed as its canonical integer view instead.
    pub fn with_view<R>(&self, consumer: impl FnOnce(SKTargetView<'_, F>) -> R) -> R {
        match self {
            SKTargetExtraction::Continuous(values) => {
                consumer(SKTargetView::Continuous(values.view()))
            }
            SKTargetExtraction::Integer(values) => consumer(SKTargetView::Integer(values.view())),
            SKTargetExtraction::Nominal {
                indices, labels, ..
            } => {
                if labels.iter().all(Option::is_some) {
                    let borrowed: Vec<&str> =
                        labels.iter().filter_map(|label| label.as_deref()).collect();
                    consumer(SKTargetView::Nominal(&borrowed))
                } else {
                    consumer(SKTargetView::Integer(indices.view()))
                }
            }
        }
    }

    /// Convert the target to owned model-scalar values.
    ///
    /// # Errors
    /// Returns [`SKError::Conversion`] when an integer target is outside the
    /// scalar's exact range.
    pub fn to_scalar_values(&self) -> Result<Array1<F>, SKError> {
        match self {
            SKTargetExtraction::Continuous(values) => Ok(values.clone()),
            SKTargetExtraction::Integer(values) => values
                .iter()
                .map(|value| {
                    cast::<i64, F>(*value).ok_or_else(|| {
                        SKError::Conversion(format!(
                            "target value {value} is not representable in the model scalar"
                        ))
                    })
                })
                .collect(),
            SKTargetExtraction::Nominal { indices, .. } => indices
                .iter()
                .map(|value| {
                    cast::<i64, F>(*value).ok_or_else(|| {
                        SKError::Conversion(format!(
                            "target index {value} is not representable in the model scalar"
                        ))
                    })
                })
                .collect(),
        }
    }
}
