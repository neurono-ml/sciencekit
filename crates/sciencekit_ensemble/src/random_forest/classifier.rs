//! The random-forest classifier configuration estimator.

use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKError, SKExecutionContext, SKExecutionMode, SKExecutionPlan,
    SKFloat,
};
use sciencekit_tree::{SKClassWeight, SKMaximumFeatures, SKTreeCriterion};

use crate::bootstrap::SKMaximumSamples;

use super::classifier_builder::SKRandomForestClassifierBuilder;

/// Breiman random-forest classifier, constructed only through its builder.
#[derive(Debug, Clone)]
pub struct SKRandomForestClassifier<F> {
    pub(super) plan: SKBuilderState,
    pub(super) number_of_trees: usize,
    pub(super) criterion: SKTreeCriterion,
    pub(super) maximum_depth: Option<usize>,
    pub(super) minimum_samples_split: usize,
    pub(super) minimum_samples_leaf: usize,
    pub(super) maximum_features: SKMaximumFeatures<F>,
    pub(super) bootstrap: bool,
    pub(super) maximum_samples: SKMaximumSamples<F>,
    pub(super) random_seed: u64,
    pub(super) class_weight: SKClassWeight<F>,
    pub(super) oob_score: bool,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKRandomForestClassifier<F> {
    /// Start building a random-forest classifier (the only construction surface).
    pub fn build() -> SKRandomForestClassifierBuilder<F> {
        SKRandomForestClassifierBuilder::new()
    }

    /// The configured number of trees in the forest.
    pub fn number_of_trees(&self) -> usize {
        self.number_of_trees
    }

    /// The split criterion passed to every tree.
    pub fn criterion(&self) -> SKTreeCriterion {
        self.criterion
    }

    /// The maximum tree depth, or `None` when unlimited.
    pub fn maximum_depth(&self) -> Option<usize> {
        self.maximum_depth
    }

    /// The minimum number of samples required to split an internal node.
    pub fn minimum_samples_split(&self) -> usize {
        self.minimum_samples_split
    }

    /// The minimum number of samples required in each leaf.
    pub fn minimum_samples_leaf(&self) -> usize {
        self.minimum_samples_leaf
    }

    /// The per-node feature-subsampling policy.
    pub fn maximum_features(&self) -> &SKMaximumFeatures<F> {
        &self.maximum_features
    }

    /// Whether each tree draws a bootstrap sample with replacement.
    pub fn bootstrap(&self) -> bool {
        self.bootstrap
    }

    /// The per-tree sample-size policy.
    pub fn maximum_samples(&self) -> &SKMaximumSamples<F> {
        &self.maximum_samples
    }

    /// The global seed from which per-tree seeds are derived.
    pub fn random_seed(&self) -> u64 {
        self.random_seed
    }

    /// The class-weighting policy passed to every tree.
    pub fn class_weight(&self) -> &SKClassWeight<F> {
        &self.class_weight
    }

    /// Whether out-of-bag accuracy estimation is enabled.
    pub fn oob_score(&self) -> bool {
        self.oob_score
    }

    /// The backend kind recorded in operation spans and transferred to models.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// The configured execution intent.
    pub fn execution_intent(&self) -> SKExecutionMode {
        self.plan.execution_intent()
    }

    /// Resolve the execution intent against an execution context.
    pub fn resolve_plan(&self, context: &SKExecutionContext) -> Result<SKExecutionPlan, SKError> {
        self.plan.resolve_plan(context)
    }
}
