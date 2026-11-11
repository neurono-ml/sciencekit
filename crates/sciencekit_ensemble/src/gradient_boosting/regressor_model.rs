//! Fitted gradient-boosting regressor model and its accessors.

use ndarray::ArrayView1;
use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};
use sciencekit_tree::SKTree;

use super::loss_kind::SKGradientBoostingRegressionLoss;

/// Fitted stage-wise additive gradient-boosting regressor.
#[derive(Debug, Clone)]
pub struct SKGradientBoostingRegressorModel<F> {
    pub(super) stages: Vec<SKTree<F>>,
    pub(super) initial_prediction: F,
    pub(super) learning_rate: F,
    pub(super) feature_count: usize,
    pub(super) feature_importances: Vec<F>,
    pub(super) loss: SKGradientBoostingRegressionLoss,
    pub(super) huber_delta: F,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
}

impl<F: SKFloat> SKGradientBoostingRegressorModel<F> {
    /// Number of boosting stages actually fitted (after any early stopping).
    pub fn number_of_stages(&self) -> usize {
        self.stages.len()
    }

    /// Borrow all fitted stage trees in sequential stage order.
    pub fn stage_trees(&self) -> &[SKTree<F>] {
        &self.stages
    }

    /// Borrow the stage tree grown at `stage_index`.
    pub fn stage_tree(&self, stage_index: usize) -> Option<&SKTree<F>> {
        self.stages.get(stage_index)
    }

    /// Constant initial prediction added before the stage trees.
    pub fn initial_prediction(&self) -> F {
        self.initial_prediction
    }

    /// Shrinkage factor applied to every stage's Newton leaf values.
    pub fn learning_rate(&self) -> F {
        self.learning_rate
    }

    /// Number of input features the booster was fitted on.
    pub fn feature_count(&self) -> usize {
        self.feature_count
    }

    /// Sample-weighted normalized feature importances aggregated across stages.
    pub fn feature_importances(&self) -> ArrayView1<'_, F> {
        ArrayView1::from(&self.feature_importances)
    }

    /// Configured regression loss carried from the estimator.
    pub fn loss(&self) -> SKGradientBoostingRegressionLoss {
        self.loss
    }

    /// Huber clipping threshold `delta` carried from the estimator.
    pub fn huber_delta(&self) -> F {
        self.huber_delta
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
