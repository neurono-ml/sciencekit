//! Shared input validation for the ranking metrics.

use std::collections::BTreeSet;

use ndarray::ArrayView2;
use sciencekit_common::{SKError, SKFloat};

/// Reject empty, length-mismatched and non-finite score/label inputs.
pub(crate) fn sk_validate_score_label_inputs<F: SKFloat>(
    scores: &[F],
    labels: &[i64],
    operation: &'static str,
) -> Result<(), SKError> {
    if labels.is_empty() || scores.is_empty() {
        return Err(SKError::EmptyInput { operation });
    }
    if scores.len() != labels.len() {
        return Err(SKError::shape_mismatch_2d(labels.len(), 1, scores.len(), 1));
    }
    for (position, score) in scores.iter().enumerate() {
        if !score.is_finite() {
            return Err(SKError::NonFiniteValue { position });
        }
    }
    Ok(())
}

/// Count the positive and negative observations of a binary problem.
pub(crate) fn sk_binary_class_counts(
    labels: &[i64],
    positive_label: i64,
    operation: &'static str,
) -> Result<(usize, usize), SKError> {
    let positives = labels
        .iter()
        .filter(|label| **label == positive_label)
        .count();
    let negatives = labels.len() - positives;
    if positives == 0 || negatives == 0 {
        return Err(SKError::UnsupportedOperation {
            operation,
            reason: format!(
                "the metric requires both classes; found {positives} positive and {negatives} negative observations"
            ),
        });
    }
    Ok((positives, negatives))
}

/// Require at least one positive observation.
pub(crate) fn sk_require_positive_class(
    labels: &[i64],
    positive_label: i64,
    operation: &'static str,
) -> Result<usize, SKError> {
    let positives = labels
        .iter()
        .filter(|label| **label == positive_label)
        .count();
    if positives == 0 {
        return Err(SKError::UnsupportedOperation {
            operation,
            reason: "the metric requires at least one positive observation".to_string(),
        });
    }
    Ok(positives)
}

/// The indices of a slice ordered by increasing score.
pub(crate) fn sk_ascending_score_order<F: SKFloat>(scores: &[F]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..scores.len()).collect();
    order.sort_by(|left, right| {
        scores[*left]
            .partial_cmp(&scores[*right])
            .expect("scores are finite")
    });
    order
}

/// Validate a multiclass score matrix against its labels and return the
/// ascending class labels.
pub(crate) fn sk_validate_multiclass_inputs<F: SKFloat>(
    scores: &ArrayView2<'_, F>,
    labels: &[i64],
    operation: &'static str,
) -> Result<Vec<i64>, SKError> {
    if labels.is_empty() {
        return Err(SKError::EmptyInput { operation });
    }
    if scores.nrows() != labels.len() {
        return Err(SKError::shape_mismatch_2d(
            labels.len(),
            scores.ncols(),
            scores.nrows(),
            scores.ncols(),
        ));
    }
    if scores.ncols() == 0 {
        return Err(SKError::shape_mismatch_2d(labels.len(), 1, labels.len(), 0));
    }
    for ((row, column), score) in scores.indexed_iter() {
        if !score.is_finite() {
            return Err(SKError::NonFiniteValue {
                position: row * scores.ncols() + column,
            });
        }
    }
    let classes: Vec<i64> = labels
        .iter()
        .copied()
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    if classes.len() < 2 {
        return Err(SKError::UnsupportedOperation {
            operation,
            reason: "the metric requires at least two classes".to_string(),
        });
    }
    if scores.ncols() != classes.len() {
        return Err(SKError::shape_mismatch_2d(
            labels.len(),
            classes.len(),
            scores.nrows(),
            scores.ncols(),
        ));
    }
    Ok(classes)
}
