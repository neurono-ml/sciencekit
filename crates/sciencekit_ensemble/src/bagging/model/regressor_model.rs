//! Fitted generic bagging regressor model.

use sciencekit_common::{SKBackendKind, SKError, SKExecutionMode, SKFloat};

use super::oob_state::SKBaggingOobState;

/// Fitted generic bagging regressor over base model `M`.
#[derive(Debug, Clone)]
pub struct SKBaggingRegressorModel<F, M> {
    pub(in crate::bagging) estimators: Vec<M>,
    pub(in crate::bagging) feature_subsets: Vec<Vec<usize>>,
    pub(in crate::bagging) out_of_bag_rows: Option<Vec<Vec<usize>>>,
    pub(in crate::bagging) oob_state: SKBaggingOobState<F>,
    pub(in crate::bagging) feature_count: usize,
    pub(in crate::bagging) bootstrap: bool,
    pub(in crate::bagging) bootstrap_features: bool,
    pub(in crate::bagging) backend: SKBackendKind,
    pub(in crate::bagging) execution_intent: SKExecutionMode,
}

impl<F: SKFloat, M> SKBaggingRegressorModel<F, M> {
    /// The number of fitted base estimators in the ensemble.
    pub fn number_of_estimators(&self) -> usize {
        self.estimators.len()
    }

    /// The number of features the ensemble was fitted on.
    pub fn feature_count(&self) -> usize {
        self.feature_count
    }

    /// Every fitted base model in deterministic slot order.
    pub fn estimators(&self) -> &[M] {
        &self.estimators
    }

    /// Borrow the underlying base model at `estimator_index`.
    pub fn estimator(&self, estimator_index: usize) -> Option<&M> {
        self.estimators.get(estimator_index)
    }

    /// Every estimator's routed feature column indices in slot order.
    pub fn feature_subsets(&self) -> &[Vec<usize>] {
        &self.feature_subsets
    }

    /// The routed feature column indices used by `estimator_index`.
    pub fn estimator_features(&self, estimator_index: usize) -> Option<&[usize]> {
        self.feature_subsets.get(estimator_index).map(Vec::as_slice)
    }

    /// The out-of-bag R² score, or `UnsupportedOperation` when unavailable.
    pub fn oob_score(&self) -> Result<F, SKError> {
        self.oob_state.score()
    }

    /// The number of training rows covered by at least one out-of-bag estimator.
    pub fn oob_scored_rows(&self) -> usize {
        self.oob_state.scored_rows()
    }

    /// The training row indices absent from `estimator_index`'s bootstrap sample
    /// when OOB scoring was enabled.
    pub fn out_of_bag_rows(&self, estimator_index: usize) -> Option<&[usize]> {
        self.out_of_bag_rows
            .as_ref()
            .and_then(|all| all.get(estimator_index).map(Vec::as_slice))
    }

    /// Whether rows were sampled with replacement at fit time.
    pub fn bootstrap(&self) -> bool {
        self.bootstrap
    }

    /// Whether features were sampled with replacement at fit time.
    pub fn bootstrap_features(&self) -> bool {
        self.bootstrap_features
    }

    /// The backend kind carried from the estimator.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// The execution intent carried from the estimator.
    pub fn execution_intent(&self) -> SKExecutionMode {
        self.execution_intent
    }
}
