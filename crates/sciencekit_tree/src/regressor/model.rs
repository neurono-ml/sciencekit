//! The fitted decision-tree regressor.

use ndarray::ArrayView1;
use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};

use crate::criterion::SKTreeCriterion;
use crate::node::SKTree;

/// Fitted CART regression tree: the node array and the normalized feature
/// importances.
#[derive(Debug, Clone)]
pub struct SKDecisionTreeRegressorModel<F> {
    pub(super) tree: SKTree<F>,
    pub(super) feature_importances: Vec<F>,
    pub(super) criterion: SKTreeCriterion,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
}

impl<F: SKFloat> SKDecisionTreeRegressorModel<F> {
    /// The fitted tree.
    pub fn tree(&self) -> &SKTree<F> {
        &self.tree
    }

    /// The number of nodes.
    pub fn node_count(&self) -> usize {
        self.tree.node_count()
    }

    /// The depth of the tree.
    pub fn depth(&self) -> usize {
        self.tree.depth()
    }

    /// The number of features the model was fitted on.
    pub fn feature_count(&self) -> usize {
        self.tree.feature_count()
    }

    /// Normalized impurity-based feature importances.
    pub fn feature_importances(&self) -> ArrayView1<'_, F> {
        ArrayView1::from(&self.feature_importances)
    }

    /// The split criterion.
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
