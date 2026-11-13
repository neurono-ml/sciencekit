//! Fitted classifier successive-halving grid search model
//! (spec `halving-grid-search`, requirement `Refit and prediction`).

use sciencekit_common::{
    SKBackendKind, SKError, SKExecutionMode, SKExportable, SKFloat, SKImportable, SKModelState,
    SKSupervisedFit,
};

use crate::grid_search::SKClassifierScoring;
use crate::halving::SKHalvingRound;

/// Fitted classifier halving search holding the elimination trace, winning
/// candidate configuration and optional refitted winner.
pub struct SKHalvingGridSearchClassifierModel<F: SKFloat, E: SKSupervisedFit<F>> {
    pub(crate) elimination_trace: Vec<SKHalvingRound<F>>,
    pub(crate) best_candidate_index: usize,
    pub(crate) best_score: F,
    pub(crate) best_candidate: E,
    pub(crate) best_model: Option<E::Model>,
    pub(crate) scoring: SKClassifierScoring,
    pub(crate) execution_mode: SKExecutionMode,
    pub(crate) backend: SKBackendKind,
}

impl<F: SKFloat, E: SKSupervisedFit<F>> SKHalvingGridSearchClassifierModel<F, E> {
    /// Round-by-round successive-halving elimination trace.
    pub fn elimination_trace(&self) -> &[SKHalvingRound<F>] {
        &self.elimination_trace
    }

    /// Number of rounds executed during the search.
    pub fn number_of_rounds(&self) -> usize {
        self.elimination_trace.len()
    }

    /// Evaluated candidate counts per round followed by the final survivor count.
    pub fn candidate_count_trajectory(&self) -> Vec<usize> {
        let mut counts: Vec<usize> = self
            .elimination_trace
            .iter()
            .map(|round| round.evaluated_candidate_indices().len())
            .collect();
        if let Some(last) = self.elimination_trace.last() {
            counts.push(last.surviving_candidate_indices().len());
        }
        counts
    }

    /// Resource budget evaluated at each executed round.
    pub fn resource_schedule(&self) -> Vec<usize> {
        self.elimination_trace
            .iter()
            .map(SKHalvingRound::resource)
            .collect()
    }

    /// Index of the winning candidate in the configured candidate slice.
    pub fn best_candidate_index(&self) -> usize {
        self.best_candidate_index
    }

    /// Mean cross-validated score of the winner in the final round.
    pub fn best_score(&self) -> F {
        self.best_score
    }

    /// Winning candidate configuration.
    pub fn best_parameters(&self) -> &E {
        &self.best_candidate
    }

    /// Winning model refitted on all rows when `refit(true)` was configured.
    pub fn best_model(&self) -> Option<&E::Model> {
        self.best_model.as_ref()
    }

    /// Whether a refitted model is stored.
    pub fn is_refitted(&self) -> bool {
        self.best_model.is_some()
    }

    /// Classification scoring policy used during the search.
    pub fn scoring(&self) -> SKClassifierScoring {
        self.scoring
    }

    /// Execution intent carried into prediction.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.execution_mode
    }

    /// Backend kind recorded in every model operation span.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }
}

impl<F, E> SKExportable<F> for SKHalvingGridSearchClassifierModel<F, E>
where
    F: SKFloat,
    E: SKSupervisedFit<F>,
    E::Model: SKExportable<F>,
{
    fn export_state(&self) -> Result<SKModelState<F>, SKError> {
        let model = self
            .best_model
            .as_ref()
            .ok_or_else(|| SKError::UnsupportedOperation {
                operation: "export_state",
                reason: "the halving grid search was built with refit(false), so no best model is stored".to_owned(),
            })?;
        model.export_state()
    }
}

impl<F, E> SKImportable<F> for SKHalvingGridSearchClassifierModel<F, E>
where
    F: SKFloat,
    E: SKSupervisedFit<F>,
{
    fn import_state(_state: &SKModelState<F>) -> Result<Self, SKError> {
        Err(SKError::UnsupportedOperation {
            operation: "import_state",
            reason: "a successive-halving state carries the refitted winner's tensors but not \
                     the candidate configuration or elimination trace; re-fit the search instead"
                .to_string(),
        })
    }
}
