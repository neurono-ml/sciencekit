//! The configuration accumulator shared by both grid-search builders.

use ndarray::Array1;
use sciencekit_common::{
    SKBackendKind, SKBuilderState, SKError, SKFloat, sk_validate_hyperparameter,
};

use super::super::configuration::SKGridSearchConfiguration;
use super::super::error_score::SKErrorScore;
use super::super::scoring::{SKScoringFamily, sk_validate_scoring_family};
use crate::SKSplitter;

/// Accumulated grid-search configuration before family-specific validation.
pub(crate) struct SKGridSearchBuildState<F: SKFloat, E> {
    pub(crate) builder_state: SKBuilderState,
    pub(crate) candidates: Vec<E>,
    pub(crate) splitter: Option<Box<dyn SKSplitter<F> + Send + Sync>>,
    pub(crate) refit: bool,
    pub(crate) error_score: SKErrorScore<F>,
    pub(crate) groups: Option<Array1<i64>>,
    pub(crate) backend: SKBackendKind,
}

impl<F: SKFloat, E> SKGridSearchBuildState<F, E> {
    /// Documented defaults: no candidates, no splitter, refit on, raise on
    /// failure, `Automatic` execution and the `Faer` backend.
    pub(crate) fn new() -> Self {
        Self {
            builder_state: SKBuilderState::new(),
            candidates: Vec::new(),
            splitter: None,
            refit: true,
            error_score: SKErrorScore::Raise,
            groups: None,
            backend: SKBackendKind::Faer,
        }
    }

    pub(crate) fn candidates(mut self, candidates: Vec<E>) -> Self {
        self.candidates = candidates;
        self
    }

    pub(crate) fn splitter<S>(mut self, splitter: S) -> Self
    where
        S: SKSplitter<F> + Send + Sync + 'static,
    {
        self.splitter = Some(Box::new(splitter));
        self
    }

    pub(crate) fn refit(mut self, refit: bool) -> Self {
        self.refit = refit;
        self
    }

    pub(crate) fn error_score(mut self, error_score: SKErrorScore<F>) -> Self {
        self.error_score = error_score;
        self
    }

    pub(crate) fn groups(mut self, groups: Array1<i64>) -> Self {
        self.groups = Some(groups);
        self
    }

    /// Validates the accumulated configuration and freezes it.
    pub(crate) fn build(
        self,
        found_scoring_family: SKScoringFamily,
        expected_scoring_family: SKScoringFamily,
    ) -> Result<SKGridSearchConfiguration<F, E>, SKError> {
        sk_validate_hyperparameter(
            "candidates",
            !self.candidates.is_empty(),
            "at least one configured candidate is required",
        )?;
        sk_validate_scoring_family(found_scoring_family, expected_scoring_family)?;
        let splitter = self.splitter.ok_or(SKError::InvalidHyperparameter {
            name: "splitter",
            reason: "a splitter is required to produce the cross-validation folds".to_owned(),
        })?;
        Ok(SKGridSearchConfiguration {
            candidates: self.candidates,
            splitter,
            refit: self.refit,
            error_score: self.error_score,
            groups: self.groups,
            execution_mode: self.builder_state.execution_intent(),
            backend: self.backend,
        })
    }
}

impl<F: SKFloat, E> Default for SKGridSearchBuildState<F, E> {
    fn default() -> Self {
        Self::new()
    }
}
