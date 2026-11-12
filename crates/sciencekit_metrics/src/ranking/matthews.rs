//! The Matthews correlation coefficient (MCC).
//!
//! The binary coefficient generalises to any number of classes through the
//! confusion-matrix covariance form used by scikit-learn; a zero denominator
//! yields zero (the documented policy).

use sciencekit_common::{SKError, SKFloat};

use crate::classification::{SKConfusionMatrix, sk_confusion_matrix};

/// Compute the Matthews correlation coefficient of matched label pairs.
///
/// The value lies in `[-1, 1]`; a zero denominator (a degenerate confusion
/// matrix) yields zero rather than NaN.
pub fn sk_matthews_correlation_coefficient<F: SKFloat>(
    true_labels: &[i64],
    predicted_labels: &[i64],
) -> Result<F, SKError> {
    let matrix = sk_confusion_matrix(true_labels, predicted_labels, None)?;
    Ok(sk_matthews_from_matrix(&matrix))
}

/// The generalised MCC over a confusion matrix.
fn sk_matthews_from_matrix<F: SKFloat>(matrix: &SKConfusionMatrix) -> F {
    let classes = matrix.number_of_classes();
    let observations = num_traits::cast::<usize, F>(matrix.number_of_observations())
        .expect("counts fit in the scalar");
    let mut correct = F::zero();
    let mut true_squares = F::zero();
    let mut predicted_squares = F::zero();
    let mut cross_product = F::zero();
    for index in 0..classes {
        let true_total =
            num_traits::cast::<usize, F>(matrix.support(index)).expect("counts fit in the scalar");
        let predicted_total = num_traits::cast::<usize, F>(matrix.prediction_count(index))
            .expect("counts fit in the scalar");
        correct = correct
            + num_traits::cast::<usize, F>(matrix.true_positive(index))
                .expect("counts fit in the scalar");
        true_squares = true_squares + true_total * true_total;
        predicted_squares = predicted_squares + predicted_total * predicted_total;
        cross_product = cross_product + true_total * predicted_total;
    }

    let covariance = correct * observations - cross_product;
    let predicted_variance = observations * observations - predicted_squares;
    let true_variance = observations * observations - true_squares;
    let denominator = predicted_variance * true_variance;
    if denominator.is_zero() || covariance.is_zero() {
        return F::zero();
    }
    covariance / denominator.sqrt()
}
