//! Fitted random-forest classifier model and per-tree slot representation.

use ndarray::ArrayView1;
use sciencekit_common::{SKBackendKind, SKError, SKExecutionMode, SKFloat, SKLabelTable};
use sciencekit_tree::{SKDecisionTreeClassifierModel, SKTree, SKTreeCriterion};

/// One fitted member of a classification forest.
///
/// Most bootstrap samples carry at least two classes and hold a full CART tree
/// plus the mapping from its local class order to the forest's canonical class
/// table. When a small bootstrap draw lands entirely in one class, CART has
/// zero impurity and cannot split, so `PureLeaf` stores that constant class
/// directly.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(super) enum SKClassifierTreeMember<F> {
    Fitted {
        model: SKDecisionTreeClassifierModel<F>,
        local_to_forest: Vec<usize>,
    },
    PureLeaf {
        class_index: usize,
    },
}

/// Fitted out-of-bag state stored on a random-forest model.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub(super) enum SKForestOobState<F> {
    Disabled,
    NoScoredRows,
    Scored { score: F, scored_rows: usize },
}

impl<F: Copy> SKForestOobState<F> {
    pub(super) fn score(&self) -> Result<F, SKError> {
        match self {
            SKForestOobState::Scored { score, .. } => Ok(*score),
            SKForestOobState::Disabled => Err(SKError::UnsupportedOperation {
                operation: "oob_score",
                reason: "out-of-bag scoring was not enabled on the builder".to_string(),
            }),
            SKForestOobState::NoScoredRows => Err(SKError::UnsupportedOperation {
                operation: "oob_score",
                reason: "no training rows fell out of bag across the fitted trees".to_string(),
            }),
        }
    }

    pub(super) fn scored_rows(&self) -> usize {
        match self {
            SKForestOobState::Scored { scored_rows, .. } => *scored_rows,
            SKForestOobState::Disabled | SKForestOobState::NoScoredRows => 0,
        }
    }
}

/// Fitted Breiman random-forest classifier.
#[derive(Debug, Clone)]
pub struct SKRandomForestClassifierModel<F> {
    pub(super) members: Vec<SKClassifierTreeMember<F>>,
    pub(super) out_of_bag_rows: Option<Vec<Vec<usize>>>,
    pub(super) oob_state: SKForestOobState<F>,
    pub(super) classes: SKLabelTable,
    pub(super) number_of_classes: usize,
    pub(super) feature_count: usize,
    pub(super) feature_importances: Vec<F>,
    pub(super) criterion: SKTreeCriterion,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
}

impl<F: SKFloat> SKRandomForestClassifierModel<F> {
    /// The number of fitted trees in the forest.
    pub fn number_of_trees(&self) -> usize {
        self.members.len()
    }

    /// The number of features the forest was fitted on.
    pub fn feature_count(&self) -> usize {
        self.feature_count
    }

    /// The canonical reversible class table learned from the full targets.
    pub fn classes(&self) -> &SKLabelTable {
        &self.classes
    }

    /// The number of canonical classes.
    pub fn number_of_classes(&self) -> usize {
        self.number_of_classes
    }

    /// Normalized forest feature importances.
    pub fn feature_importances(&self) -> ArrayView1<'_, F> {
        ArrayView1::from(&self.feature_importances)
    }

    /// The out-of-bag accuracy, or `UnsupportedOperation` when unavailable.
    pub fn oob_score(&self) -> Result<F, SKError> {
        self.oob_state.score()
    }

    /// The number of training rows covered by at least one out-of-bag tree.
    pub fn oob_scored_rows(&self) -> usize {
        self.oob_state.scored_rows()
    }

    /// The training row indices absent from `tree_index`'s bootstrap sample
    /// when OOB scoring was enabled.
    pub fn out_of_bag_rows(&self, tree_index: usize) -> Option<&[usize]> {
        self.out_of_bag_rows
            .as_ref()
            .and_then(|all| all.get(tree_index).map(Vec::as_slice))
    }

    /// Inspect the underlying CART node structure of `tree_index` when that
    /// bootstrap sample carried at least two classes.
    pub fn tree(&self, tree_index: usize) -> Option<&SKTree<F>> {
        match self.members.get(tree_index)? {
            SKClassifierTreeMember::Fitted { model, .. } => Some(model.tree()),
            SKClassifierTreeMember::PureLeaf { .. } => None,
        }
    }

    /// The split criterion carried from the estimator.
    pub fn criterion(&self) -> SKTreeCriterion {
        self.criterion
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
