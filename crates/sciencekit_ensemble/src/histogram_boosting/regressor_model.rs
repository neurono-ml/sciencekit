//! Fitted histogram gradient-boosting regressor and its accessors.

use ndarray::ArrayView1;
use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat};

use super::binning::SKFeatureBinning;
use super::growth::SKHistogramTree;
use super::loss_kind::SKHistGradientBoostingRegressionLoss;

/// Fitted histogram gradient-boosting regressor.
#[derive(Debug, Clone)]
pub struct SKHistGradientBoostingRegressorModel<F> {
    pub(super) stages: Vec<SKHistogramTree<F>>,
    pub(super) stage_importances: Vec<Vec<F>>,
    pub(super) initial_prediction: F,
    pub(super) learning_rate: F,
    pub(super) feature_count: usize,
    pub(super) feature_importances: Vec<F>,
    pub(super) feature_binnings: Vec<SKFeatureBinning<F>>,
    pub(super) loss: SKHistGradientBoostingRegressionLoss,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
}

impl<F: SKFloat> SKHistGradientBoostingRegressorModel<F> {
    /// Number of boosting stages actually fitted (after any early stopping).
    pub fn number_of_stages(&self) -> usize {
        self.stages.len()
    }

    /// Borrow all fitted stage trees in sequential stage order.
    pub fn stage_trees(&self) -> &[SKHistogramTree<F>] {
        &self.stages
    }

    /// Borrow the stage tree grown at `stage_index`.
    pub fn stage_tree(&self, stage_index: usize) -> Option<&SKHistogramTree<F>> {
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

    /// Stored quantile bin edges of a numeric feature.
    pub fn bin_edges(&self, feature_index: usize) -> Option<&[F]> {
        self.feature_binnings
            .get(feature_index)
            .map(SKFeatureBinning::edges)
    }

    /// Stored category codes of a categorical feature.
    pub fn categories(&self, feature_index: usize) -> Option<&[i64]> {
        self.feature_binnings
            .get(feature_index)
            .map(SKFeatureBinning::categories)
    }

    /// Sample-weighted normalized feature importances aggregated across stages.
    pub fn feature_importances(&self) -> ArrayView1<'_, F> {
        ArrayView1::from(&self.feature_importances)
    }

    /// Normalized feature importances of the tree grown at `stage_index`.
    pub fn stage_feature_importances(&self, stage_index: usize) -> Option<ArrayView1<'_, F>> {
        self.stage_importances
            .get(stage_index)
            .map(ArrayView1::from)
    }

    /// Configured regression loss carried from the estimator.
    pub fn loss(&self) -> SKHistGradientBoostingRegressionLoss {
        self.loss
    }

    /// Backend kind carried from the estimator and recorded in spans.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// Execution intent carried from the estimator.
    pub fn execution_intent(&self) -> SKExecutionMode {
        self.execution_intent
    }
}
