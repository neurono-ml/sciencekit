//! Fitted histogram gradient-boosting classifier and its accessors.

use ndarray::ArrayView1;
use sciencekit_common::{SKBackendKind, SKExecutionMode, SKFloat, SKLabelTable};

use super::binning::SKFeatureBinning;
use super::growth::SKHistogramTree;

/// Fitted histogram gradient-boosting classifier.
#[derive(Debug, Clone)]
pub struct SKHistGradientBoostingClassifierModel<F> {
    pub(super) stages: Vec<Vec<SKHistogramTree<F>>>,
    pub(super) stage_importances: Vec<Vec<Vec<F>>>,
    pub(super) initial_predictions: Vec<F>,
    pub(super) learning_rate: F,
    pub(super) classes: SKLabelTable,
    pub(super) number_of_classes: usize,
    pub(super) feature_count: usize,
    pub(super) feature_importances: Vec<F>,
    pub(super) feature_binnings: Vec<SKFeatureBinning<F>>,
    pub(super) backend: SKBackendKind,
    pub(super) execution_intent: SKExecutionMode,
}

impl<F: SKFloat> SKHistGradientBoostingClassifierModel<F> {
    /// Number of boosting stages actually fitted (after any early stopping).
    pub fn number_of_stages(&self) -> usize {
        self.stages.len()
    }

    /// Number of trees fitted per stage (`1` binary, `C` multiclass).
    pub fn trees_per_stage(&self) -> usize {
        if self.number_of_classes <= 2 {
            1
        } else {
            self.number_of_classes
        }
    }

    /// Borrow the trees grown at `stage_index`.
    pub fn stage_trees(&self, stage_index: usize) -> Option<&[SKHistogramTree<F>]> {
        self.stages.get(stage_index).map(Vec::as_slice)
    }

    /// Borrow `tree_index` within `stage_index`.
    pub fn stage_tree(
        &self,
        stage_index: usize,
        tree_index: usize,
    ) -> Option<&SKHistogramTree<F>> {
        self.stages
            .get(stage_index)
            .and_then(|trees| trees.get(tree_index))
    }

    /// Initial raw-score vector (`[log_odds]` for binary, `C` log-priors for
    /// multiclass).
    pub fn initial_predictions(&self) -> ArrayView1<'_, F> {
        ArrayView1::from(&self.initial_predictions)
    }

    /// Scalar initial log-odds for a binary problem.
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

    /// Stored quantile bin edges of a numeric feature (empty when categorical).
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

    /// Whether `feature_index` was treated as categorical.
    pub fn is_categorical(&self, feature_index: usize) -> Option<bool> {
        self.feature_binnings
            .get(feature_index)
            .map(SKFeatureBinning::is_categorical)
    }

    /// Sample-weighted normalized feature importances aggregated across stages.
    pub fn feature_importances(&self) -> ArrayView1<'_, F> {
        ArrayView1::from(&self.feature_importances)
    }

    /// Normalized importances of `tree_index` in `stage_index`.
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

    /// Backend kind carried from the estimator.
    pub fn backend(&self) -> SKBackendKind {
        self.backend
    }

    /// Execution intent carried from the estimator.
    pub fn execution_intent(&self) -> SKExecutionMode {
        self.execution_intent
    }
}
