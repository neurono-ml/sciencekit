//! Binary and multiclass log loss with documented clipping.
//!
//! Every probability is clipped to `[F::epsilon(), 1 − F::epsilon()]` before
//! the logarithm, so a true-class probability of exactly zero or one still
//! yields a finite loss. Probabilities outside `[0, 1]` are rejected instead of
//! silently clipped.

use ndarray::{ArrayView1, ArrayView2};
use sciencekit_common::{SKError, SKFloat};

/// Compute the multiclass log loss of a probability matrix.
///
/// Each row is one observation; `probabilities[[row, column]]` is the
/// probability of the class at `column`. When `class_labels` is absent the
/// columns map to the labels `0..columns`; otherwise the list is sorted
/// internally and must be unique and as long as the column count.
pub fn sk_log_loss<F: SKFloat>(
    true_labels: &[i64],
    probabilities: ArrayView2<'_, F>,
    class_labels: Option<&[i64]>,
) -> Result<F, SKError> {
    let operation = "log_loss";
    if true_labels.is_empty() {
        return Err(SKError::EmptyInput { operation });
    }
    let columns = probabilities.ncols();
    if probabilities.nrows() != true_labels.len() {
        return Err(SKError::shape_mismatch_2d(
            true_labels.len(),
            columns,
            probabilities.nrows(),
            columns,
        ));
    }
    if columns < 2 {
        return Err(SKError::shape_mismatch_2d(
            true_labels.len(),
            2,
            probabilities.nrows(),
            columns,
        ));
    }
    let labels = sk_resolve_class_labels(class_labels, columns)?;

    let epsilon = F::epsilon();
    let upper = F::one() - epsilon;
    let mut total = F::zero();
    for (row, label) in true_labels.iter().enumerate() {
        let column = labels
            .binary_search(label)
            .map_err(|_| SKError::UnknownLabel { label: *label })?;
        for selected in 0..columns {
            let position = row * columns + selected;
            let probability = probabilities[[row, selected]];
            if !probability.is_finite() {
                return Err(SKError::NonFiniteValue { position });
            }
            if probability < F::zero() || probability > F::one() {
                return Err(SKError::ProbabilityOutOfRange { position });
            }
        }
        total = total + sk_clipped_log(probabilities[[row, column]], epsilon, upper);
    }
    let count = num_traits::cast::<usize, F>(true_labels.len()).expect("counts fit in the scalar");
    Ok(-total / count)
}

/// Compute the binary log loss from positive-class probabilities.
pub fn sk_binary_log_loss<F: SKFloat>(
    true_labels: &[i64],
    positive_probabilities: ArrayView1<'_, F>,
    positive_label: i64,
) -> Result<F, SKError> {
    let operation = "binary_log_loss";
    if true_labels.is_empty() {
        return Err(SKError::EmptyInput { operation });
    }
    if positive_probabilities.len() != true_labels.len() {
        return Err(SKError::shape_mismatch_2d(
            true_labels.len(),
            1,
            positive_probabilities.len(),
            1,
        ));
    }
    let epsilon = F::epsilon();
    let upper = F::one() - epsilon;
    let mut total = F::zero();
    for (position, (label, probability)) in true_labels
        .iter()
        .zip(positive_probabilities.iter())
        .enumerate()
    {
        if !probability.is_finite() {
            return Err(SKError::NonFiniteValue { position });
        }
        if *probability < F::zero() || *probability > F::one() {
            return Err(SKError::ProbabilityOutOfRange { position });
        }
        let complement = sk_clipped_log(F::one() - *probability, epsilon, upper);
        let probability = sk_clipped_log(*probability, epsilon, upper);
        total = total
            + if *label == positive_label {
                probability
            } else {
                complement
            };
    }
    let count = num_traits::cast::<usize, F>(true_labels.len()).expect("counts fit in the scalar");
    Ok(-total / count)
}

/// Resolve the class-label list of a probability matrix.
fn sk_resolve_class_labels(
    class_labels: Option<&[i64]>,
    columns: usize,
) -> Result<Vec<i64>, SKError> {
    match class_labels {
        Some(labels) => {
            if labels.len() != columns {
                return Err(SKError::shape_mismatch_2d(columns, 1, labels.len(), 1));
            }
            let mut sorted = labels.to_vec();
            sorted.sort_unstable();
            for pair in sorted.windows(2) {
                if pair[0] == pair[1] {
                    return Err(SKError::InvalidHyperparameter {
                        name: "class_labels",
                        reason: format!("label {} appears more than once", pair[0]),
                    });
                }
            }
            Ok(sorted)
        }
        None => Ok((0..columns)
            .map(|column| num_traits::cast::<usize, i64>(column).expect("columns fit in i64"))
            .collect()),
    }
}

/// The natural logarithm of a clipped probability.
fn sk_clipped_log<F: SKFloat>(probability: F, epsilon: F, upper: F) -> F {
    let clipped = if probability < epsilon {
        epsilon
    } else if probability > upper {
        upper
    } else {
        probability
    };
    clipped.ln()
}
