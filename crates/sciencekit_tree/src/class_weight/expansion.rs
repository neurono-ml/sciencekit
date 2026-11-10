//! The per-sample expansion of the class-weighting policy.

use sciencekit_common::{SKError, SKFloat};

use super::policy::SKClassWeight;

/// Expand a class-weighting policy into one non-negative weight per sample.
///
/// `None` weighs every sample one; `Balanced` weighs each class by
/// `n_samples / (n_classes · class_count)`; `Explicit` uses a supplied vector
/// in canonical class order. A wrong-length, negative or all-zero explicit
/// vector is rejected with a structured error.
pub fn sk_expand_class_weights<F: SKFloat>(
    class_weight: &SKClassWeight<F>,
    class_indices: &[usize],
    number_of_classes: usize,
) -> Result<Vec<F>, SKError> {
    let per_class = match class_weight {
        SKClassWeight::None => vec![F::one(); number_of_classes],
        SKClassWeight::Balanced => {
            let mut counts = vec![0usize; number_of_classes];
            for &index in class_indices {
                counts[index] += 1;
            }
            let number_of_samples = class_indices.len();
            counts
                .iter()
                .map(|&count| {
                    if count == 0 {
                        F::zero()
                    } else {
                        F::from(number_of_samples).expect("sample counts fit")
                            / F::from(number_of_classes * count).expect("counts fit")
                    }
                })
                .collect()
        }
        SKClassWeight::Explicit(weights) => {
            if weights.len() != number_of_classes {
                return Err(SKError::ShapeMismatch {
                    expected: vec![number_of_classes],
                    found: vec![weights.len()],
                });
            }
            if weights.iter().any(|weight| *weight < F::zero()) {
                return Err(SKError::InvalidHyperparameter {
                    name: "class_weight",
                    reason: "explicit class weights must be non-negative".to_string(),
                });
            }
            if weights.iter().all(|weight| *weight == F::zero()) {
                return Err(SKError::InvalidHyperparameter {
                    name: "class_weight",
                    reason: "at least one class weight must be positive".to_string(),
                });
            }
            weights.clone()
        }
    };
    Ok(class_indices
        .iter()
        .map(|&index| per_class[index])
        .collect())
}
