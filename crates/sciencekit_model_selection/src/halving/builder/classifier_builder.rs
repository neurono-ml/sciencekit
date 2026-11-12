//! Builder-only construction for [`SKHalvingGridSearchClassifier`]
//! (spec `halving-grid-search`, requirement `Builder-only construction with documented defaults`).

use ndarray::Array1;
use sciencekit_common::{SKBackendKind, SKBuilder, SKError, SKExecutionMode, SKFloat};

use crate::SKSplitter;
use crate::grid_search::{SKClassifierScoring, SKErrorScore};
use crate::halving::SKHalvingResource;
use crate::halving::classifier::SKHalvingGridSearchClassifier;
use crate::halving::configuration::SKHalvingScoring;

use super::state::SKHalvingGridSearchBuildState;

/// Builder for [`SKHalvingGridSearchClassifier`]; the only construction path.
pub struct SKHalvingGridSearchClassifierBuilder<F: SKFloat, E> {
    state: SKHalvingGridSearchBuildState<F, E>,
    scoring: SKHalvingScoring,
}

impl<F: SKFloat, E> SKHalvingGridSearchClassifierBuilder<F, E> {
    /// Creates a builder with the documented defaults.
    pub fn new() -> Self {
        Self {
            state: SKHalvingGridSearchBuildState::new(),
            scoring: SKHalvingScoring::Classifier(SKClassifierScoring::Accuracy),
        }
    }

    /// Sets the configured candidate estimators.
    pub fn candidates(mut self, candidates: Vec<E>) -> Self {
        self.state.candidates = candidates;
        self
    }

    /// Sets the cross-validation splitter.
    pub fn splitter<S>(mut self, splitter: S) -> Self
    where
        S: SKSplitter<F> + Send + Sync + 'static,
    {
        self.state.splitter = Some(Box::new(splitter));
        self
    }

    /// Sets the scoring policy (`Accuracy` by default).
    pub fn scoring<S: Into<SKHalvingScoring>>(mut self, scoring: S) -> Self {
        self.scoring = scoring.into();
        self
    }

    /// Sets the halving elimination factor (`3` by default; must be at least `2`).
    pub fn factor(mut self, factor: usize) -> Self {
        self.state.factor = factor;
        self
    }

    /// Sets the initial resource budget (`1` by default).
    pub fn minimum_resource(mut self, minimum_resource: usize) -> Self {
        self.state.minimum_resource = minimum_resource;
        self
    }

    /// Caps the maximum resource budget (defaults to all samples when omitted).
    pub fn maximum_resource(mut self, maximum_resource: usize) -> Self {
        self.state.maximum_resource = Some(maximum_resource);
        self
    }

    /// Sets the resource dimension scaled across rounds (`NumberOfSamples` by default).
    pub fn resource(mut self, resource: SKHalvingResource) -> Self {
        self.state.resource = resource;
        self
    }

    /// Enables or disables aggressive early elimination (`false` by default).
    pub fn aggressive_elimination(mut self, aggressive_elimination: bool) -> Self {
        self.state.aggressive_elimination = aggressive_elimination;
        self
    }

    /// Sets the deterministic subsampling seed (`0` by default).
    pub fn random_seed(mut self, random_seed: u64) -> Self {
        self.state.random_seed = random_seed;
        self
    }

    /// Enables or disables refitting the winner on all rows (`true` by default).
    pub fn refit(mut self, refit: bool) -> Self {
        self.state.refit = refit;
        self
    }

    /// Sets the candidate failure policy (`SKErrorScore::Raise` by default).
    pub fn error_score(mut self, error_score: SKErrorScore<F>) -> Self {
        self.state.error_score = error_score;
        self
    }

    /// Supplies group labels read by group-aware splitters.
    pub fn groups(mut self, groups: Array1<i64>) -> Self {
        self.state.groups = Some(groups);
        self
    }

    /// Sets the backend kind recorded in operation spans (`Faer` by default).
    pub fn backend(mut self, backend: SKBackendKind) -> Self {
        self.state.backend = backend;
        self
    }

    /// Sets the execution intent (ergonomic by-value form).
    pub fn execution_mode<M>(mut self, mode: M) -> Self
    where
        M: TryInto<SKExecutionMode>,
    {
        self.state.builder_state.execution_mode(mode);
        self
    }
}

impl<F: SKFloat, E> Default for SKHalvingGridSearchClassifierBuilder<F, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat, E> SKBuilder<SKHalvingGridSearchClassifier<F, E>>
    for SKHalvingGridSearchClassifierBuilder<F, E>
{
    fn execution_mode<M>(&mut self, mode: M) -> &mut Self
    where
        M: TryInto<SKExecutionMode>,
    {
        self.state.builder_state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKHalvingGridSearchClassifier<F, E>, SKError> {
        let scoring = self.scoring.into_classifier()?;
        let configuration = self.state.build()?;
        Ok(SKHalvingGridSearchClassifier {
            configuration,
            scoring,
        })
    }
}
