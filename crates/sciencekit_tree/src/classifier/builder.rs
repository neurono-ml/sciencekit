//! The classifier configuration surface: the builder.

use sciencekit_common::{
    SKBackendKind, SKBuilder, SKBuilderState, SKError, SKExecutionMode, SKFloat,
    sk_validate_hyperparameter,
};

use crate::class_weight::SKClassWeight;
use crate::criterion::SKTreeCriterion;
use crate::splitter::{SKMaximumFeatures, SKSplitter};

use super::core_implementation::SKDecisionTreeClassifier;

/// Builder for [`SKDecisionTreeClassifier`]: the only construction surface.
pub struct SKDecisionTreeClassifierBuilder<F> {
    pub(super) state: SKBuilderState,
    pub(super) criterion: SKTreeCriterion,
    pub(super) maximum_depth: Option<usize>,
    pub(super) minimum_samples_split: usize,
    pub(super) minimum_samples_leaf: usize,
    pub(super) maximum_features: SKMaximumFeatures<F>,
    pub(super) splitter: SKSplitter,
    pub(super) random_seed: u64,
    pub(super) class_weight: SKClassWeight<F>,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKDecisionTreeClassifierBuilder<F> {
    /// A fresh builder with the documented defaults.
    pub fn new() -> Self {
        Self {
            state: SKBuilderState::new(),
            criterion: SKTreeCriterion::Gini,
            maximum_depth: None,
            minimum_samples_split: 2,
            minimum_samples_leaf: 1,
            maximum_features: SKMaximumFeatures::All,
            splitter: SKSplitter::Best,
            random_seed: 0,
            class_weight: SKClassWeight::None,
            backend: SKBackendKind::Faer,
        }
    }

    /// The split criterion (default `Gini`).
    pub fn criterion(mut self, criterion: SKTreeCriterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// Stop growing beyond this depth (default unlimited).
    pub fn maximum_depth(mut self, maximum_depth: usize) -> Self {
        self.maximum_depth = Some(maximum_depth);
        self
    }

    /// Require at least this many samples to split a node (default 2).
    pub fn minimum_samples_split(mut self, minimum_samples_split: usize) -> Self {
        self.minimum_samples_split = minimum_samples_split;
        self
    }

    /// Require at least this many samples in each leaf (default 1).
    pub fn minimum_samples_leaf(mut self, minimum_samples_leaf: usize) -> Self {
        self.minimum_samples_leaf = minimum_samples_leaf;
        self
    }

    /// The feature-subsampling policy (default `All`).
    pub fn maximum_features(mut self, maximum_features: SKMaximumFeatures<F>) -> Self {
        self.maximum_features = maximum_features;
        self
    }

    /// The split-selection strategy (default `Best`).
    pub fn splitter(mut self, splitter: SKSplitter) -> Self {
        self.splitter = splitter;
        self
    }

    /// The seed of every stochastic choice (default 0).
    pub fn random_seed(mut self, random_seed: u64) -> Self {
        self.random_seed = random_seed;
        self
    }

    /// The class-weighting policy (default `None`).
    pub fn class_weight(mut self, class_weight: SKClassWeight<F>) -> Self {
        self.class_weight = class_weight;
        self
    }

    /// The backend kind carried into every span and the fitted model
    /// (default [`SKBackendKind::Faer`]).
    pub fn backend(mut self, backend: SKBackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Set the execution intent (default `Automatic`).
    pub fn execution_mode<E>(mut self, mode: E) -> Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }
}

impl<F: SKFloat> Default for SKDecisionTreeClassifierBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat> SKBuilder<SKDecisionTreeClassifier<F>> for SKDecisionTreeClassifierBuilder<F> {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKDecisionTreeClassifier<F>, SKError> {
        sk_validate_common_configuration(
            self.criterion,
            self.minimum_samples_split,
            self.minimum_samples_leaf,
            &self.maximum_features,
        )?;
        sk_validate_hyperparameter(
            "criterion",
            is_classification(self.criterion),
            "the classifier supports Gini and Entropy",
        )?;
        sk_validate_class_weight(&self.class_weight)?;
        Ok(SKDecisionTreeClassifier {
            plan: self.state,
            criterion: self.criterion,
            maximum_depth: self.maximum_depth,
            minimum_samples_split: self.minimum_samples_split,
            minimum_samples_leaf: self.minimum_samples_leaf,
            maximum_features: self.maximum_features,
            splitter: self.splitter,
            random_seed: self.random_seed,
            class_weight: self.class_weight,
            backend: self.backend,
        })
    }
}

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
            weights.iter().all(|weight| *weight >= F::zero()),
            "explicit class weights must be non-negative",
        )?;
        sk_validate_hyperparameter(
            "class_weight",
            weights.iter().any(|weight| *weight > F::zero()),
            "at least one class weight must be positive",
        )?;
    }
    Ok(())
}
