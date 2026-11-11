//! Fitted generic bagging models and out-of-bag state.

use sciencekit_common::{SKBackendKind, SKError, SKExecutionMode, SKFloat, SKLabelTable};

/// Fitted out-of-bag state stored on a bagging model.
#[derive(Debug, Clone, PartialEq)]
#[allow(dead_code)]
pub(super) enum SKBaggingOobState<F> {
    Disabled,
    NoScoredRows,
    Scored { score: F, scored_rows: usize },
}

impl<F: Copy> SKBaggingOobState<F> {
    pub(super) fn score(&self) -> Result<F, SKError> {
        match self {
            SKBaggingOobState::Scored { score, .. } => Ok(*score),
            SKBaggingOobState::Disabled => Err(SKError::UnsupportedOperation {
                operation: "oob_score",
                reason: "out-of-bag scoring was not enabled on the builder".to_string(),
            }),
            SKBaggingOobState::NoScoredRows => Err(SKError::UnsupportedOperation {
                operation: "oob_score",
                reason: "no training rows fell out of bag across the fitted estimators".to_string(),
            }),
        }
    }

    pub(super) fn scored_rows(&self) -> usize {
        match self {
            SKBaggingOobState::Scored { scored_rows, .. } => *scored_rows,
            SKBaggingOobState::Disabled | SKBaggingOobState::NoScoredRows => 0,
        }
    }
}

/// One fitted member of a classification bagging ensemble.
#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(super) enum SKBaggingClassifierMember<M> {
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
    pub(super) members: Vec<SKBaggingClassifierMember<M>>,
    pub(super) feature_subsets: Vec<Vec<usize>>,
    pub(super) out_of_bag_rows: Option<Vec<Vec<usize>>>,
    pub(super) oob_state: SKBaggingOobState<F>,
    pub(super) classes: SKLabelTable,
    pub(super) number_of_classes: usize,
    pub(super) feature_count: usize,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
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

    /// The backend kind carried from the estimator.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// The execution intent carried from the estimator.
    pub fn execution_intent(&self) -> SKExecutionMode {
        self.execution_intent
    }
}

/// Fitted generic bagging regressor over base model `M`.
#[derive(Debug, Clone)]
pub struct SKBaggingRegressorModel<F, M> {
    pub(super) estimators: Vec<M>,
    pub(super) feature_subsets: Vec<Vec<usize>>,
    pub(super) out_of_bag_rows: Option<Vec<Vec<usize>>>,
    pub(super) oob_state: SKBaggingOobState<F>,
    pub(super) feature_count: usize,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
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

    /// The backend kind carried from the estimator.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// The execution intent carried from the estimator.
    pub fn execution_intent(&self) -> SKExecutionMode {
        self.execution_intent
    }
}
