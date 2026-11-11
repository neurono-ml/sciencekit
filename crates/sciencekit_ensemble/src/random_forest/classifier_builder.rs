//! Builder for [`SKRandomForestClassifier`]: the sole construction surface.

use sciencekit_common::{
    SKBackendKind, SKBuilder, SKBuilderState, SKError, SKExecutionMode, SKFloat,
};
use sciencekit_tree::{SKClassWeight, SKMaximumFeatures, SKTreeCriterion};

use crate::bootstrap::SKMaximumSamples;

use super::classifier::SKRandomForestClassifier;
use super::validation::{
    sk_validate_class_weight, sk_validate_classifier_criterion, sk_validate_common_forest_builder,
};

/// Builder for [`SKRandomForestClassifier`].
pub struct SKRandomForestClassifierBuilder<F> {
    state: SKBuilderState,
    number_of_trees: usize,
    criterion: SKTreeCriterion,
    maximum_depth: Option<usize>,
    minimum_samples_split: usize,
    minimum_samples_leaf: usize,
    maximum_features: SKMaximumFeatures<F>,
    bootstrap: bool,
    maximum_samples: SKMaximumSamples<F>,
    random_seed: u64,
    class_weight: SKClassWeight<F>,
    oob_score: bool,
    backend: SKBackendKind,
}

impl<F: SKFloat> SKRandomForestClassifierBuilder<F> {
    /// Create a builder initialized with the documented forest defaults.
    pub fn new() -> Self {
        Self {
            state: SKBuilderState::new(),
            number_of_trees: 100,
            criterion: SKTreeCriterion::Gini,
            maximum_depth: None,
            minimum_samples_split: 2,
            minimum_samples_leaf: 1,
            maximum_features: SKMaximumFeatures::Sqrt,
            bootstrap: true,
            maximum_samples: SKMaximumSamples::All,
            random_seed: 0,
            class_weight: SKClassWeight::None,
            oob_score: false,
            backend: SKBackendKind::Faer,
        }
    }

    /// Configure the number of trees in the forest (default `100`).
    pub fn number_of_trees(mut self, number_of_trees: usize) -> Self {
        self.number_of_trees = number_of_trees;
        self
    }

    /// Configure the split criterion (default `Gini`).
    pub fn criterion(mut self, criterion: SKTreeCriterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// Stop growing trees beyond this depth (default unlimited).
    pub fn maximum_depth(mut self, maximum_depth: usize) -> Self {
        self.maximum_depth = Some(maximum_depth);
        self
    }

    /// Require at least this many samples to split an internal node (default `2`).
    pub fn minimum_samples_split(mut self, minimum_samples_split: usize) -> Self {
        self.minimum_samples_split = minimum_samples_split;
        self
    }

    /// Require at least this many samples in every leaf (default `1`).
    pub fn minimum_samples_leaf(mut self, minimum_samples_leaf: usize) -> Self {
        self.minimum_samples_leaf = minimum_samples_leaf;
        self
    }

    /// Configure the per-split feature-subsampling policy (default `Sqrt`).
    pub fn maximum_features(mut self, maximum_features: SKMaximumFeatures<F>) -> Self {
        self.maximum_features = maximum_features;
        self
    }

    /// Configure whether each tree fits on a bootstrap sample (default `true`).
    pub fn bootstrap(mut self, bootstrap: bool) -> Self {
        self.bootstrap = bootstrap;
        self
    }

    /// Configure the per-tree sample-size policy (default `All`).
    pub fn maximum_samples(mut self, maximum_samples: SKMaximumSamples<F>) -> Self {
        self.maximum_samples = maximum_samples;
        self
    }

    /// Configure the global deterministic random seed (default `0`).
    pub fn random_seed(mut self, random_seed: u64) -> Self {
        self.random_seed = random_seed;
        self
    }

    /// Configure the class-weighting policy (default `None`).
    pub fn class_weight(mut self, class_weight: SKClassWeight<F>) -> Self {
        self.class_weight = class_weight;
        self
    }

    /// Configure whether out-of-bag accuracy is estimated (default `false`).
    pub fn oob_score(mut self, oob_score: bool) -> Self {
        self.oob_score = oob_score;
        self
    }

    /// Configure the backend recorded in spans and models (default `Faer`).
    pub fn backend(mut self, backend: SKBackendKind) -> Self {
        self.backend = backend;
        self
    }

    /// Configure the execution intent (default `Automatic`).
    pub fn execution_mode<E>(mut self, mode: E) -> Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }
}

impl<F: SKFloat> Default for SKRandomForestClassifierBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat> SKBuilder<SKRandomForestClassifier<F>> for SKRandomForestClassifierBuilder<F> {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKRandomForestClassifier<F>, SKError> {
        sk_validate_common_forest_builder(
            self.number_of_trees,
            self.minimum_samples_split,
            self.minimum_samples_leaf,
            &self.maximum_features,
            self.bootstrap,
            &self.maximum_samples,
            self.oob_score,
        )?;
        sk_validate_classifier_criterion(self.criterion)?;
        sk_validate_class_weight(&self.class_weight)?;
        Ok(SKRandomForestClassifier {
            plan: self.state,
            number_of_trees: self.number_of_trees,
            criterion: self.criterion,
            maximum_depth: self.maximum_depth,
            minimum_samples_split: self.minimum_samples_split,
            minimum_samples_leaf: self.minimum_samples_leaf,
            maximum_features: self.maximum_features,
            bootstrap: self.bootstrap,
            maximum_samples: self.maximum_samples,
            random_seed: self.random_seed,
            class_weight: self.class_weight,
            oob_score: self.oob_score,
            backend: self.backend,
        })
    }
}
