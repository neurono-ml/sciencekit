//! Class-weighting policy and per-class weight resolution
//! (spec `support-vector-machines`).

use sciencekit_common::{SKError, SKFloat};

/// Class-weighting policy scaling the box bound `C` of each class.
#[derive(Debug, Clone, PartialEq, Default)]
pub enum SKClassWeight<F> {
    /// Every class weighs `1`.
    #[default]
    None,
    /// Weight each class inversely to its frequency: `n_samples / (n_classes · count_c)`.
    Balanced,
    /// Explicit per-class weights in canonical class order `0..n_classes`.
    Explicit(Vec<F>),
}

/// Validate static class-weight invariants known before seeing training labels.
pub fn sk_validate_static_class_weight<F: SKFloat>(
    class_weight: &SKClassWeight<F>,
) -> Result<(), SKError> {
    if let SKClassWeight::Explicit(weights) = class_weight {
        if weights.is_empty() {
            return Err(SKError::InvalidHyperparameter {
                name: "class_weight",
                reason: "explicit class weights must not be empty".to_owned(),
            });
        }
        if weights
            .iter()
            .any(|weight| !weight.is_finite() || *weight < F::zero())
        {
            return Err(SKError::InvalidHyperparameter {
                name: "class_weight",
                reason: "explicit class weights must be finite and non-negative".to_owned(),
            });
        }
        if weights.iter().all(|weight| *weight == F::zero()) {
            return Err(SKError::InvalidHyperparameter {
                name: "class_weight",
                reason: "at least one explicit class weight must be positive".to_owned(),
            });
        }
    }
    Ok(())
}

/// Resolve per-class weights of length `number_of_classes` for `class_indices`.
pub fn sk_resolve_class_weights<F: SKFloat>(
    class_weight: &SKClassWeight<F>,
    class_indices: &[usize],
    number_of_classes: usize,
) -> Result<Vec<F>, SKError> {
    sk_validate_static_class_weight(class_weight)?;
    match class_weight {
        SKClassWeight::None => Ok(vec![F::one(); number_of_classes]),
        SKClassWeight::Balanced => {
            let mut counts = vec![0usize; number_of_classes];
            class_indices.iter().for_each(|&class| counts[class] += 1);
            let total = F::from(class_indices.len()).expect("sample count fits scalar");
            Ok(counts
                .into_iter()
                .map(|count| {
                    if count == 0 {
                        F::zero()
                    } else {
                        total / F::from(number_of_classes * count).expect("count fits scalar")
                    }
                })
                .collect())
        }
        SKClassWeight::Explicit(weights) => {
            if weights.len() != number_of_classes {
                return Err(SKError::ShapeMismatch {
                    expected: vec![number_of_classes],
                    found: vec![weights.len()],
                });
            }
            Ok(weights.clone())
        }
    }
}
