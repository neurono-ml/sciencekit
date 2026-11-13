//! Balanced accuracy: the mean per-class recall.

use sciencekit_common::{SKError, SKFloat};

use crate::classification::{SKConfusionMatrix, sk_confusion_matrix};

/// Compute balanced accuracy as the mean recall over the classes present in
/// the true labels.
///
/// Classes that only appear among the predictions have zero support and are
/// skipped, matching scikit-learn's `balanced_accuracy_score`.
pub fn sk_balanced_accuracy_score<F: SKFloat>(
    true_labels: &[i64],
    predicted_labels: &[i64],
) -> Result<F, SKError> {
    let matrix = sk_confusion_matrix(true_labels, predicted_labels, None)?;
    Ok(sk_mean_supported_recall(&matrix))
}

/// The mean recall over classes with positive true support.
fn sk_mean_supported_recall<F: SKFloat>(matrix: &SKConfusionMatrix) -> F {
    let mut total = F::zero();
    let mut supported = 0_usize;
    for index in 0..matrix.number_of_classes() {
        let support = matrix.support(index);
        if support == 0 {
            continue;
        }
        let true_positives = matrix.true_positive(index);
        total = total + sk_ratio(true_positives, support);
        supported += 1;
    }
    let supported = num_traits::cast::<usize, F>(supported).expect("counts fit in the scalar");
    total / supported
}

/// A ratio that yields zero on a zero denominator.
fn sk_ratio<F: SKFloat>(numerator: usize, denominator: usize) -> F {
    if denominator == 0 {
        return F::zero();
    }
    let numerator = num_traits::cast::<usize, F>(numerator).expect("counts fit in the scalar");
    let denominator = num_traits::cast::<usize, F>(denominator).expect("counts fit in the scalar");
    numerator / denominator
}
