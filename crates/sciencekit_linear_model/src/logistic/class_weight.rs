//! The class-weight expansion of the logistic substrate (spec
//! `logistic-solver`).

use sciencekit_common::{SKError, SKFloat};

use super::config::SKClassWeight;

/// Expand a class-weighting policy into one non-negative weight per sample,
/// indexed by the sample's canonical class index.
///
/// `None` weighs every sample one; `Balanced` weighs each class by
/// `n_samples / (n_classes · class_count)`; `Explicit` uses the supplied
/// per-class vector. A weight vector of the wrong length, a negative weight or
/// an all-zero weighting is rejected with a structured error.
pub(crate) fn sk_expand_class_weights<F: SKFloat>(
    class_weight: &SKClassWeight<F>,
    class_indices: &[usize],
    number_of_classes: usize,
) -> Result<Vec<F>, SKError> {
    let per_class = match class_weight {
        SKClassWeight::None => vec![F::one(); number_of_classes],
        SKClassWeight::Balanced => {
            let mut counts = vec![0usize; number_of_classes];
            class_indices.iter().for_each(|&index| counts[index] += 1);
            let n_samples = class_indices.len();
            counts
                .iter()
                .map(|&count| {
                    if count == 0 {
                        F::zero()
                    } else {
                        F::from(n_samples).expect("sample counts fit in the scalar")
                            / F::from(number_of_classes * count).expect("counts fit in the scalar")
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
