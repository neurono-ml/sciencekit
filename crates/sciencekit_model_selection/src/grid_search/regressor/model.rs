//! The fitted regressor grid search (spec `grid-search`).

use ndarray::{ArrayView1, ArrayView2};
use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat, SKSupervisedFit};

use crate::grid_search::results::SKGridSearchResults;
use crate::grid_search::scoring::SKRegressorScoring;

/// Fitted regressor grid search: the cross-validation results, the winning
/// candidate configuration and, when refit is enabled, the winner trained on
/// all rows.
pub struct SKGridSearchCVRegressorModel<F: SKFloat, E: SKSupervisedFit<F>> {
    pub(crate) results: SKGridSearchResults<F>,
    pub(crate) best_candidate: E,
    pub(crate) best_model: Option<E::Model>,
    pub(crate) scoring: SKRegressorScoring,
    pub(crate) execution_mode: SKExecutionMode,
    pub(crate) backend: SKBackendKind,
}

impl<F: SKFloat, E: SKSupervisedFit<F>> SKGridSearchCVRegressorModel<F, E> {
    /// The cross-validated results of every candidate.
    pub fn results(&self) -> &SKGridSearchResults<F> {
        &self.results
    }

    /// The winning candidate index.
    pub fn best_candidate_index(&self) -> usize {
        self.results.best_index()
    }

    /// The winning candidate's mean cross-validated score.
    pub fn best_score(&self) -> F {
        self.results.best_score()
    }

    /// The winning candidate configuration (the selected "parameters").
    pub fn best_parameters(&self) -> &E {
        &self.best_candidate
    }

    /// The winner refitted on all rows, when refit was enabled.
    pub fn best_model(&self) -> Option<&E::Model> {
        self.best_model.as_ref()
    }

    /// Whether the winner was refitted on all rows.
    pub fn is_refitted(&self) -> bool {
        self.best_model.is_some()
    }

    /// Per-candidate mean cross-validated scores.
    pub fn mean_scores(&self) -> ArrayView1<'_, F> {
        self.results.mean_scores()
    }

    /// Per-candidate population standard deviations.
    pub fn standard_deviations(&self) -> ArrayView1<'_, F> {
        self.results.standard_deviations()
    }

    /// The raw `(candidates × folds)` score matrix.
    pub fn fold_scores(&self) -> ArrayView2<'_, F> {
        self.results.fold_scores()
    }

    /// Candidate indices ranked by descending mean, ties by ascending index.
    pub fn ranked_candidate_indices(&self) -> Vec<usize> {
        self.results.ranked_candidate_indices()
    }

    /// The scoring policy used during the search.
    pub fn scoring(&self) -> SKRegressorScoring {
        self.scoring
    }

    /// The execution intent carried into prediction.
    pub fn execution_mode(&self) -> SKExecutionMode {
        self.execution_mode
    }

    /// The backend kind recorded by every model operation.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }
}
