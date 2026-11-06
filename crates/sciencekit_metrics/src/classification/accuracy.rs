//! Accuracy over matched label pairs.

use sciencekit_common::{SKError, SKFloat};

/// The fraction of observations whose predicted label matches the true label.
pub fn sk_accuracy_score<F: SKFloat>(
    true_labels: &[i64],
    predicted_labels: &[i64],
) -> Result<F, SKError> {
    if true_labels.is_empty() {
        return Err(SKError::EmptyInput {
            operation: "accuracy",
        });
    }
    if true_labels.len() != predicted_labels.len() {
        return Err(SKError::shape_mismatch_2d(
            true_labels.len(),
            1,
            predicted_labels.len(),
            1,
        ));
    }
    let correct = true_labels
        .iter()
        .zip(predicted_labels.iter())
        .filter(|(true_label, predicted_label)| true_label == predicted_label)
        .count();
    let correct = num_traits::cast::<usize, F>(correct).expect("counts fit in the scalar");
    let total = num_traits::cast::<usize, F>(true_labels.len()).expect("counts fit in the scalar");
    Ok(correct / total)
}
