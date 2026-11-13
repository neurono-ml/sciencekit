//! The fitted decision-tree classifier.

use ndarray::ArrayView1;
use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat, SKLabelTable};

use crate::criterion::SKTreeCriterion;
use crate::node::SKTree;

/// Fitted CART classification tree: the node array, the canonical class table
/// and the normalized feature importances.
#[derive(Debug, Clone)]
pub struct SKDecisionTreeClassifierModel<F> {
    pub(super) tree: SKTree<F>,
    pub(super) classes: SKLabelTable,
    pub(super) number_of_classes: usize,
    pub(super) feature_importances: Vec<F>,
    pub(super) criterion: SKTreeCriterion,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
}

impl<F: SKFloat> SKDecisionTreeClassifierModel<F> {
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

    /// The reversible class table.
    pub fn classes(&self) -> &SKLabelTable {
        &self.classes
    }

    /// The number of fitted classes.
    pub fn number_of_classes(&self) -> usize {
        self.number_of_classes
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

    /// The error for features whose column count disagrees with the fit.
    pub(super) fn feature_mismatch(
        &self,
        rows: usize,
        columns: usize,
    ) -> sciencekit_common::SKError {
        sciencekit_common::SKError::shape_mismatch_2d(rows, self.feature_count(), rows, columns)
    }
}
