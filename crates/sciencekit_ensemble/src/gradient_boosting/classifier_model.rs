//! Fitted gradient-boosting classifier model and its accessors.

use ndarray::ArrayView1;
use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat, SKLabelTable};
use sciencekit_tree::SKTree;

use super::loss_kind::SKGradientBoostingClassificationLoss;

/// Fitted stage-wise additive gradient-boosting classifier.
#[derive(Debug, Clone)]
pub struct SKGradientBoostingClassifierModel<F> {
    pub(super) stages: Vec<Vec<SKTree<F>>>,
    pub(super) stage_importances: Vec<Vec<Vec<F>>>,
    pub(super) initial_predictions: Vec<F>,
    pub(super) learning_rate: F,
    pub(super) classes: SKLabelTable,
    pub(super) number_of_classes: usize,
    pub(super) feature_count: usize,
    pub(super) feature_importances: Vec<F>,
    pub(super) loss: SKGradientBoostingClassificationLoss,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
}

impl<F: SKFloat> SKGradientBoostingClassifierModel<F> {
    /// Number of boosting stages actually fitted (after any early stopping).
    pub fn number_of_stages(&self) -> usize {
        self.stages.len()
    }

    /// Number of trees fitted per stage (`1` for binary, `C` for multiclass).
    pub fn trees_per_stage(&self) -> usize {
        if self.number_of_classes <= 2 {
            1
        } else {
            self.number_of_classes
        }
    }

    /// Borrow the trees grown at `stage_index`.
    pub fn stage_trees(&self, stage_index: usize) -> Option<&[SKTree<F>]> {
        self.stages.get(stage_index).map(Vec::as_slice)
    }

    /// Borrow `tree_index` within `stage_index` (`0` for binary, `0..C` for
    /// multiclass).
    pub fn stage_tree(&self, stage_index: usize, tree_index: usize) -> Option<&SKTree<F>> {
        self.stages
            .get(stage_index)
            .and_then(|trees| trees.get(tree_index))
    }

    /// Initial raw-score vector (`[log_odds]` for binary, `C` log-priors for
    /// multiclass).
    pub fn initial_predictions(&self) -> ArrayView1<'_, F> {
        ArrayView1::from(&self.initial_predictions)
    }

    /// Scalar initial log-odds when the problem is binary, or `None` for
    /// multiclass.
    pub fn initial_prediction(&self) -> Option<F> {
        (self.initial_predictions.len() == 1).then(|| self.initial_predictions[0])
    }

    /// Shrinkage factor applied to every stage's Newton leaf values.
    pub fn learning_rate(&self) -> F {
        self.learning_rate
    }

    /// Canonical reversible class table learned during fit.
    pub fn classes(&self) -> &SKLabelTable {
        &self.classes
    }

    /// Number of canonical classes `C`.
    pub fn number_of_classes(&self) -> usize {
        self.number_of_classes
    }

    /// Number of input features the booster was fitted on.
    pub fn feature_count(&self) -> usize {
        self.feature_count
    }

    /// Sample-weighted normalized feature importances aggregated across stages.
    pub fn feature_importances(&self) -> ArrayView1<'_, F> {
        ArrayView1::from(&self.feature_importances)
    }

    /// Normalized feature importances of `tree_index` in `stage_index`.
    pub fn stage_feature_importances(
        &self,
        stage_index: usize,
        tree_index: usize,
    ) -> Option<ArrayView1<'_, F>> {
        self.stage_importances
            .get(stage_index)
            .and_then(|trees| trees.get(tree_index))
            .map(ArrayView1::from)
    }

    /// Configured classification loss carried from the estimator.
    pub fn loss(&self) -> SKGradientBoostingClassificationLoss {
        self.loss
    }

    /// Backend kind carried from the estimator and recorded in prediction spans.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// Execution intent carried from the estimator.
    pub fn execution_intent(&self) -> SKExecutionMode {
        self.execution_intent
    }
}
