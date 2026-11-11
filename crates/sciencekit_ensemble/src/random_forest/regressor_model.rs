//! Fitted random-forest regressor model.

use ndarray::ArrayView1;
use sciencekit_common::{SKBackendKind, SKError, SKExecutionMode, SKFloat};
use sciencekit_tree::{SKDecisionTreeRegressorModel, SKTree, SKTreeCriterion};

use super::classifier_model::SKForestOobState;

/// Fitted Breiman random-forest regressor.
#[derive(Debug, Clone)]
pub struct SKRandomForestRegressorModel<F> {
    pub(super) trees: Vec<SKDecisionTreeRegressorModel<F>>,
    pub(super) out_of_bag_rows: Option<Vec<Vec<usize>>>,
    pub(super) oob_state: SKForestOobState<F>,
    pub(super) feature_count: usize,
    pub(super) feature_importances: Vec<F>,
    pub(super) criterion: SKTreeCriterion,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
}

impl<F: SKFloat> SKRandomForestRegressorModel<F> {
    /// The number of fitted trees in the forest.
    pub fn number_of_trees(&self) -> usize {
        self.trees.len()
    }

    /// The number of features the forest was fitted on.
    pub fn feature_count(&self) -> usize {
        self.feature_count
    }

    /// Every fitted regression tree in deterministic slot order.
    pub fn trees(&self) -> &[SKDecisionTreeRegressorModel<F>] {
        &self.trees
    }

    /// Inspect the underlying CART node structure of `tree_index`.
    pub fn tree(&self, tree_index: usize) -> Option<&SKTree<F>> {
        self.trees.get(tree_index).map(SKDecisionTreeRegressorModel::tree)
    }

    /// Normalized forest feature importances.
    pub fn feature_importances(&self) -> ArrayView1<'_, F> {
        ArrayView1::from(&self.feature_importances)
    }

    /// The out-of-bag R² score, or `UnsupportedOperation` when unavailable.
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
