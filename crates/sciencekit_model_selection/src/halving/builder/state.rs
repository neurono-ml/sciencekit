//! Shared builder accumulator for classifier and regressor halving searches.

use ndarray::Array1;
use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKError, SKFloat, sk_validate_hyperparameter,
};

use crate::SKSplitter;
use crate::grid_search::SKErrorScore;
use crate::halving::SKHalvingResource;
use crate::halving::configuration::SKHalvingGridSearchConfiguration;

pub(crate) struct SKHalvingGridSearchBuildState<F: SKFloat, E> {
    pub(crate) builder_state: SKBuilderState,
    pub(crate) candidates: Vec<E>,
    pub(crate) splitter: Option<Box<dyn SKSplitter<F> + Send + Sync>>,
    pub(crate) factor: usize,
    pub(crate) minimum_resource: usize,
    pub(crate) maximum_resource: Option<usize>,
    pub(crate) resource: SKHalvingResource,
    pub(crate) aggressive_elimination: bool,
    pub(crate) random_seed: u64,
    pub(crate) refit: bool,
    pub(crate) error_score: SKErrorScore<F>,
    pub(crate) groups: Option<Array1<i64>>,
    pub(crate) backend: SKBackendKind,
}

impl<F: SKFloat, E> SKHalvingGridSearchBuildState<F, E> {
    pub(crate) fn new() -> Self {
        Self {
            builder_state: SKBuilderState::new(),
            candidates: Vec::new(),
            splitter: None,
            factor: 3,
            minimum_resource: 1,
            maximum_resource: None,
            resource: SKHalvingResource::NumberOfSamples,
            aggressive_elimination: false,
            random_seed: 0,
            refit: true,
            error_score: SKErrorScore::Raise,
            groups: None,
            backend: SKBackendKind::Faer,
        }
    }

    pub(crate) fn build(self) -> Result<SKHalvingGridSearchConfiguration<F, E>, SKError> {
        sk_validate_hyperparameter(
            "candidates",
            !self.candidates.is_empty(),
            "at least one configured candidate is required",
        )?;
        sk_validate_hyperparameter(
            "factor",
            self.factor >= 2,
            "the halving factor must be at least 2",
        )?;
        sk_validate_hyperparameter(
            "minimum_resource",
            self.minimum_resource >= 1,
            "minimum_resource must be at least 1",
        )?;
        if let Some(maximum_resource) = self.maximum_resource {
            sk_validate_hyperparameter(
                "maximum_resource",
                maximum_resource >= 1,
                "maximum_resource must be at least 1",
            )?;
            sk_validate_hyperparameter(
                "minimum_resource",
                self.minimum_resource <= maximum_resource,
                "minimum_resource must not exceed maximum_resource",
            )?;
        }
        let splitter = self.splitter.ok_or(SKError::InvalidHyperparameter {
            name: "splitter",
            reason: "a splitter is required to produce the cross-validation folds".to_owned(),
        })?;

        Ok(SKHalvingGridSearchConfiguration {
            candidates: self.candidates,
            splitter,
            factor: self.factor,
            minimum_resource: self.minimum_resource,
            maximum_resource: self.maximum_resource,
            resource: self.resource,
            aggressive_elimination: self.aggressive_elimination,
            random_seed: self.random_seed,
            refit: self.refit,
            error_score: self.error_score,
            groups: self.groups,
            execution_mode: self.builder_state.execution_intent(),
            backend: self.backend,
        })
    }
}
