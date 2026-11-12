//! Configuration and input validation shared by both random-forest estimators.

use sciencekit_common::{SKError, SKFloat, sk_validate_hyperparameter};
use sciencekit_tree::{SKClassWeight, SKMaximumFeatures, SKTreeCriterion};

use crate::bootstrap::SKMaximumSamples;

/// Validate the hyperparameters shared by both forest builders.
pub(super) fn sk_validate_common_forest_builder<F: SKFloat>(
    number_of_trees: usize,
    minimum_samples_split: usize,
    minimum_samples_leaf: usize,
    maximum_features: &SKMaximumFeatures<F>,
    bootstrap: bool,
    maximum_samples: &SKMaximumSamples<F>,
    oob_score: bool,
) -> Result<(), SKError> {
    sk_validate_hyperparameter(
        "number_of_trees",
        number_of_trees >= 1,
        "number_of_trees must be at least 1",
    )?;
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
        )?,
        SKMaximumFeatures::Fraction(fraction) => sk_validate_hyperparameter(
            "maximum_features",
            fraction.is_finite() && *fraction > F::zero() && *fraction <= F::one(),
            "a feature fraction must be finite and in (0, 1]",
        )?,
        _ => {}
    }
    match maximum_samples {
        SKMaximumSamples::All => {}
        SKMaximumSamples::Count(count) => sk_validate_hyperparameter(
            "maximum_samples",
            *count >= 1,
            "an explicit sample count must be at least 1",
        )?,
        SKMaximumSamples::Fraction(fraction) => sk_validate_hyperparameter(
            "maximum_samples",
            fraction.is_finite() && *fraction > F::zero() && *fraction <= F::one(),
            "a sample fraction must be finite and in (0, 1]",
        )?,
    }
    sk_validate_hyperparameter(
        "oob_score",
        !oob_score || bootstrap,
        "out-of-bag scoring requires bootstrap sampling to be enabled",
    )
}

/// Validate that `criterion` belongs to the classification family.
pub(super) fn sk_validate_classifier_criterion(criterion: SKTreeCriterion) -> Result<(), SKError> {
    sk_validate_hyperparameter(
        "criterion",
        matches!(criterion, SKTreeCriterion::Gini | SKTreeCriterion::Entropy),
        "the random-forest classifier supports Gini and Entropy",
    )
}

/// Validate that `criterion` belongs to the regression family.
pub(super) fn sk_validate_regressor_criterion(criterion: SKTreeCriterion) -> Result<(), SKError> {
    sk_validate_hyperparameter(
        "criterion",
        matches!(
            criterion,
            SKTreeCriterion::SquaredError | SKTreeCriterion::AbsoluteError
        ),
        "the random-forest regressor supports SquaredError and AbsoluteError",
    )
}

/// Validate an explicit class-weight vector at build time.
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
