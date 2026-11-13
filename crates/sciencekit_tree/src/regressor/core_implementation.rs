//! The `SKDecisionTreeRegressor` estimator.

use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKError, SKExecutionContext, SKExecutionMode, SKExecutionPlan,
    SKFloat, SKSupervisedFit,
};

use crate::criterion::SKTreeCriterion;
use crate::splitter::{SKMaximumFeatures, SKSplitter};

use super::builder::SKDecisionTreeRegressorBuilder;
use super::model::SKDecisionTreeRegressorModel;

/// CART regression tree, builder-configured.
///
/// Immutable configuration; the learned state lives only on the distinct
/// [`SKDecisionTreeRegressorModel`].
#[derive(Debug, Clone)]
pub struct SKDecisionTreeRegressor<F> {
    pub(super) plan: SKBuilderState,
    pub(super) criterion: SKTreeCriterion,
    pub(super) maximum_depth: Option<usize>,
    pub(super) minimum_samples_split: usize,
    pub(super) minimum_samples_leaf: usize,
    pub(super) maximum_features: SKMaximumFeatures<F>,
    pub(super) splitter: SKSplitter,
    pub(super) random_seed: u64,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKDecisionTreeRegressor<F> {
    /// Start building an estimator (the only construction surface).
    pub fn build() -> SKDecisionTreeRegressorBuilder<F> {
        SKDecisionTreeRegressorBuilder::new()
    }

    /// The configured execution intent.
    pub fn execution_intent(&self) -> SKExecutionMode {
        self.plan.execution_intent()
    }

    /// Resolve the execution intent against a context.
    pub fn resolve_plan(&self, context: &SKExecutionContext) -> Result<SKExecutionPlan, SKError> {
        self.plan.resolve_plan(context)
    }

    /// The backend recorded in operation spans and transferred to fitted models.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// The split criterion.
    pub fn criterion(&self) -> SKTreeCriterion {
        self.criterion
    }

    /// The maximum depth, or `None` when unlimited.
    pub fn maximum_depth(&self) -> Option<usize> {
        self.maximum_depth
    }

    /// The minimum number of samples a node must hold to be split.
    pub fn minimum_samples_split(&self) -> usize {
        self.minimum_samples_split
    }

    /// The minimum number of samples each leaf must hold.
    pub fn minimum_samples_leaf(&self) -> usize {
        self.minimum_samples_leaf
    }

    /// The feature-subsampling policy.
    pub fn maximum_features(&self) -> &SKMaximumFeatures<F> {
        &self.maximum_features
    }

    /// The split-selection strategy.
    pub fn splitter(&self) -> SKSplitter {
        self.splitter
    }

    /// The seed of every stochastic choice.
    pub fn random_seed(&self) -> u64 {
        self.random_seed
    }
}

impl<F: SKFloat> SKSupervisedFit<F> for SKDecisionTreeRegressor<F> {
    type Model = SKDecisionTreeRegressorModel<F>;
    type Error = SKError;

    fn fit<'a, X, T>(&self, features: X, targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<sciencekit_common::SKDataView<'a, F>, Error = SKError>,
        T: TryInto<sciencekit_common::SKTargetView<'a, F>, Error = SKError>,
    {
        super::fit::sk_fit_regressor(self, features.try_into()?, targets.try_into()?)
    }
}
