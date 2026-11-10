//! Builder-only construction for `SKGridSearchCVRegressor` (spec
//! `grid-search`, requirement `Builder-only construction with documented
//! defaults`).

use ndarray::Array1;
use sciencekit_common::{SKBackendKind, SKBuilder, SKError, SKExecutionMode, SKFloat};

use super::state::SKGridSearchBuildState;
use crate::SKSplitter;
use crate::grid_search::error_score::SKErrorScore;
use crate::grid_search::regressor::SKGridSearchCVRegressor;
use crate::grid_search::scoring::{SKRegressorScoring, SKScoringFamily};

/// Builder for [`SKGridSearchCVRegressor`]; the only construction path.
///
/// Defaults: refit enabled, `SKErrorScore::Raise`, `Automatic` execution and
/// the `Faer` backend.
pub struct SKGridSearchCVRegressorBuilder<F: SKFloat, E> {
    state: SKGridSearchBuildState<F, E>,
    scoring: SKRegressorScoring,
}

impl<F: SKFloat, E> SKGridSearchCVRegressorBuilder<F, E> {
    /// A builder carrying the documented defaults.
    pub fn new() -> Self {
        Self {
            state: SKGridSearchBuildState::new(),
            scoring: SKRegressorScoring::NegativeMeanSquaredError,
        }
    }

    /// Sets the fully configured candidates to evaluate.
    pub fn candidates(mut self, candidates: Vec<E>) -> Self {
        self.state = self.state.candidates(candidates);
        self
    }

    /// Sets the splitter producing the cross-validation folds.
    pub fn splitter<S>(mut self, splitter: S) -> Self
    where
        S: SKSplitter<F> + Send + Sync + 'static,
    {
        self.state = self.state.splitter(splitter);
        self
    }

    /// Sets the regression scoring policy (negative MSE by default).
    pub fn scoring(mut self, scoring: SKRegressorScoring) -> Self {
        self.scoring = scoring;
        self
    }

    /// Enables or disables refitting the winner on all rows (`true` by
    /// default).
    pub fn refit(mut self, refit: bool) -> Self {
        self.state = self.state.refit(refit);
        self
    }

    /// Sets the failing-fold policy (`SKErrorScore::Raise` by default).
    pub fn error_score(mut self, error_score: SKErrorScore<F>) -> Self {
        self.state = self.state.error_score(error_score);
        self
    }

    /// Supplies the group vector read by a group-aware splitter.
    pub fn groups(mut self, groups: Array1<i64>) -> Self {
        self.state = self.state.groups(groups);
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

impl<F: SKFloat, E> Default for SKGridSearchCVRegressorBuilder<F, E> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SKFloat, E> SKBuilder<SKGridSearchCVRegressor<F, E>>
    for SKGridSearchCVRegressorBuilder<F, E>
{
    fn execution_mode<M>(&mut self, mode: M) -> &mut Self
    where
        M: TryInto<SKExecutionMode>,
    {
        self.state.builder_state.execution_mode(mode);
        self
    }

    fn build(self) -> Result<SKGridSearchCVRegressor<F, E>, SKError> {
        let scoring = self.scoring;
        let configuration = self
            .state
            .build(scoring.family(), SKScoringFamily::Regressor)?;
        Ok(SKGridSearchCVRegressor {
            configuration,
            scoring,
        })
    }
}
