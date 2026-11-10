//! Shared classifier configuration validation.

use sciencekit_common::{SKError, SKFloat, sk_validate_hyperparameter};

use crate::class_weight::SKClassWeight;
use crate::criterion::SKTreeCriterion;
use crate::splitter::SKMaximumFeatures;

pub(super) fn is_classification(criterion: SKTreeCriterion) -> bool {
    matches!(criterion, SKTreeCriterion::Gini | SKTreeCriterion::Entropy)
}

pub(crate) fn sk_validate_common_configuration<F: SKFloat>(
    _criterion: SKTreeCriterion,
    minimum_samples_split: usize,
    minimum_samples_leaf: usize,
    maximum_features: &SKMaximumFeatures<F>,
) -> Result<(), SKError> {
    sk_validate_hyperparameter(
        "minimum_samples_split",
        minimum_samples_split >= 2,
        "minimum_samples_split must be at least 2",
    )?;
    sk_validate_hyperparameter(
        "minimum_samples_leaf",
        minimum_samples_leaf >= 1,
        "minimum_samples_leaf must be at least 1",
    )?;
    match maximum_features {
        SKMaximumFeatures::Count(count) => sk_validate_hyperparameter(
            "maximum_features",
            *count >= 1,
            "an explicit feature count must be at least 1",
        ),
        SKMaximumFeatures::Fraction(fraction) => sk_validate_hyperparameter(
            "maximum_features",
            *fraction > F::zero() && *fraction <= F::one(),
            "a feature fraction must be in (0, 1]",
        ),
        _ => Ok(()),
    }
}

pub(super) fn sk_validate_class_weight<F: SKFloat>(
    class_weight: &SKClassWeight<F>,
) -> Result<(), SKError> {
    if let SKClassWeight::Explicit(weights) = class_weight {
        sk_validate_hyperparameter(
            "class_weight",
            !weights.is_empty(),
            "an explicit class-weight vector must not be empty",
        )?;
        sk_validate_hyperparameter(
            "class_weight",
            weights
                .iter()
                .all(|weight| weight.is_finite() && *weight >= F::zero()),
            "explicit class weights must be finite and non-negative",
        )?;
        sk_validate_hyperparameter(
            "class_weight",
            weights.iter().any(|weight| *weight > F::zero()),
            "at least one class weight must be positive",
        )?;
    }
    Ok(())
}
