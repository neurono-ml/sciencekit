//! Fitted generic bagging classifier model.

use sciencekit_common::{SKBackendKind, SKError, SKExecutionMode, SKFloat, SKLabelTable};

use super::oob_state::SKBaggingOobState;

/// One fitted member of a classification bagging ensemble.
#[derive(Debug, Clone)]
pub(in crate::bagging) enum SKBaggingClassifierMember<M> {
    Fitted {
        model: M,
        local_to_ensemble: Vec<usize>,
    },
    ConstantClass {
        class_index: usize,
    },
}

/// Fitted generic bagging classifier over base model `M`.
#[derive(Debug, Clone)]
pub struct SKBaggingClassifierModel<F, M> {
    pub(in crate::bagging) members: Vec<SKBaggingClassifierMember<M>>,
    pub(in crate::bagging) feature_subsets: Vec<Vec<usize>>,
    pub(in crate::bagging) out_of_bag_rows: Option<Vec<Vec<usize>>>,
    pub(in crate::bagging) oob_state: SKBaggingOobState<F>,
    pub(in crate::bagging) classes: SKLabelTable,
    pub(in crate::bagging) number_of_classes: usize,
    pub(in crate::bagging) feature_count: usize,
    pub(in crate::bagging) bootstrap: bool,
    pub(in crate::bagging) bootstrap_features: bool,
    pub(in crate::bagging) backend: SKBackendKind,
    pub(in crate::bagging) execution_intent: SKExecutionMode,
}

impl<F: SKFloat, M> SKBaggingClassifierModel<F, M> {
    /// The number of fitted base members in the ensemble.
    pub fn number_of_estimators(&self) -> usize {
        self.members.len()
    }

    /// The number of features the ensemble was fitted on.
    pub fn feature_count(&self) -> usize {
        self.feature_count
    }

    /// Every estimator's routed feature column indices in slot order.
    pub fn feature_subsets(&self) -> &[Vec<usize>] {
        &self.feature_subsets
    }

    /// The routed feature column indices used by `estimator_index`.
    pub fn estimator_features(&self, estimator_index: usize) -> Option<&[usize]> {
        self.feature_subsets.get(estimator_index).map(Vec::as_slice)
    }

    /// Borrow the underlying base model at `estimator_index` when that slot was
    /// not a constant single-class bootstrap draw.
    pub fn estimator(&self, estimator_index: usize) -> Option<&M> {
        match self.members.get(estimator_index)? {
            SKBaggingClassifierMember::Fitted { model, .. } => Some(model),
            SKBaggingClassifierMember::ConstantClass { .. } => None,
        }
    }

    /// The canonical reversible class table learned from the full targets.
    pub fn classes(&self) -> &SKLabelTable {
        &self.classes
    }

    /// The number of canonical classes.
    pub fn number_of_classes(&self) -> usize {
        self.number_of_classes
    }

    /// The out-of-bag accuracy, or `UnsupportedOperation` when unavailable.
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
