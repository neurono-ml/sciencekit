//! Positive-class score-column resolution shared by the ranking adapters.

use ndarray::ArrayView2;
use sciencekit_common::{SKError, SKFloat};

/// Extract the positive-class score column of a binary probability matrix.
pub(super) fn sk_positive_score_column<F: SKFloat>(
    probabilities: ArrayView2<'_, F>,
    positive_label: i64,
    class_labels: Option<&[i64]>,
) -> Result<Vec<F>, SKError> {
    if probabilities.ncols() != 2 {
        return Err(SKError::UnsupportedOperation {
            operation: "binary_ranking_scorer",
            reason: "binary ranking metrics require exactly two probability columns".to_string(),
        });
    }
    let column = match class_labels {
        Some(labels) => {
            if labels.len() != probabilities.ncols() {
                return Err(SKError::shape_mismatch_2d(
                    probabilities.ncols(),
                    1,
                    labels.len(),
                    1,
                ));
            }
            let mut sorted = labels.to_vec();
            sorted.sort_unstable();
            sorted
                .binary_search(&positive_label)
                .map_err(|_| SKError::UnknownLabel {
                    label: positive_label,
                })?
        }
        None => match positive_label {
            0 => 0,
            1 => 1,
            _ => {
                return Err(SKError::UnknownLabel {
                    label: positive_label,
                });
            }
        },
    };
    Ok(probabilities.column(column).to_vec())
}
