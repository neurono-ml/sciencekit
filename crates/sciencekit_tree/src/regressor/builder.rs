//! The regressor configuration surface: the builder.

use sciencekit_common::{
    SKBackendKind, SKBuilder, SKBuilderState, SKError, SKExecutionMode, SKFloat,
    sk_validate_hyperparameter,
};

use crate::criterion::SKTreeCriterion;
use crate::splitter::{SKMaximumFeatures, SKSplitter};

use super::core_implementation::SKDecisionTreeRegressor;

/// Builder for [`SKDecisionTreeRegressor`]: the only construction surface.
pub struct SKDecisionTreeRegressorBuilder<F> {
    pub(super) state: SKBuilderState,
    pub(super) criterion: SKTreeCriterion,
    pub(super) maximum_depth: Option<usize>,
    pub(super) minimum_samples_split: usize,
    pub(super) minimum_samples_leaf: usize,
    pub(super) maximum_features: SKMaximumFeatures<F>,
    pub(super) splitter: SKSplitter,
    pub(super) random_seed: u64,
    pub(super) backend: SKBackendKind,
}

impl<F: SKFloat> SKDecisionTreeRegressorBuilder<F> {
    /// A fresh builder with the documented defaults.
    pub fn new() -> Self {
        Self {
            state: SKBuilderState::new(),
            criterion: SKTreeCriterion::SquaredError,
            maximum_depth: None,
            minimum_samples_split: 2,
            minimum_samples_leaf: 1,
            maximum_features: SKMaximumFeatures::All,
            splitter: SKSplitter::Best,
            random_seed: 0,
            backend: SKBackendKind::Faer,
        }
    }

    /// The split criterion (default `SquaredError`).
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

impl<F: SKFloat> Default for SKDecisionTreeRegressorBuilder<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat> SKBuilder<SKDecisionTreeRegressor<F>> for SKDecisionTreeRegressorBuilder<F> {
    fn execution_mode<E>(&mut self, mode: E) -> &mut Self
    where
        E: TryInto<SKExecutionMode>,
    {
        self.state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKDecisionTreeRegressor<F>, SKError> {
        crate::classifier::validation::sk_validate_common_configuration(
            self.criterion,
            self.minimum_samples_split,
            self.minimum_samples_leaf,
            &self.maximum_features,
        )?;
        sk_validate_hyperparameter(
            "criterion",
            is_regression(self.criterion),
            "the regressor supports SquaredError and AbsoluteError",
        )?;
        Ok(SKDecisionTreeRegressor {
            plan: self.state,
            criterion: self.criterion,
            maximum_depth: self.maximum_depth,
            minimum_samples_split: self.minimum_samples_split,
            minimum_samples_leaf: self.minimum_samples_leaf,
            maximum_features: self.maximum_features,
            splitter: self.splitter,
            random_seed: self.random_seed,
            backend: self.backend,
        })
    }
}

fn is_regression(criterion: SKTreeCriterion) -> bool {
    matches!(
        criterion,
        SKTreeCriterion::SquaredError | SKTreeCriterion::AbsoluteError
    )
}
