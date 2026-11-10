//! Ranking and probability scorer adapters.

use ndarray::{ArrayView1, ArrayView2};
use sciencekit_common::{
    SKClassifierPredictor, SKError, SKFloat, SKLabelScorer, SKProbabilityScorer, SKTargetView,
};

use super::targets::sk_labels_from_targets;
use crate::ranking::{
    sk_average_precision_score, sk_balanced_accuracy_score, sk_log_loss,
    sk_matthews_correlation_coefficient, sk_roc_auc_score,
};

/// The binary AUC of a classifier's probabilities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SKAucScorer {
    positive_label: i64,
    class_labels: Option<Vec<i64>>,
}

impl SKAucScorer {
    /// Build an AUC scorer for the chosen positive class.
    pub fn new(positive_label: i64) -> Self {
        Self {
            positive_label,
            class_labels: None,
        }
    }

    /// Pin the class-label list that maps probability columns to labels.
    pub fn with_class_labels(mut self, labels: &[i64]) -> Self {
        self.class_labels = Some(labels.to_vec());
        self
    }
}

impl<F: SKFloat, M: SKClassifierPredictor<F>> SKProbabilityScorer<F, M> for SKAucScorer
where
    SKError: From<M::Error>,
{
    type Error = SKError;

    fn score_from_probabilities(
        &self,
        true_targets: SKTargetView<'_, F>,
        probabilities: ArrayView2<'_, F>,
    ) -> Result<F, SKError> {
        let truth = sk_labels_from_targets(&true_targets)?;
        let scores = sk_positive_score_column(
            probabilities,
            self.positive_label,
            self.class_labels.as_deref(),
        )?;
        sk_roc_auc_score(
            &scores,
            truth.as_slice().expect("an owned array is contiguous"),
            self.positive_label,
        )
    }
}

/// The binary average precision of a classifier's probabilities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SKAveragePrecisionScorer {
    positive_label: i64,
    class_labels: Option<Vec<i64>>,
}

impl SKAveragePrecisionScorer {
    /// Build an average-precision scorer for the chosen positive class.
    pub fn new(positive_label: i64) -> Self {
        Self {
            positive_label,
            class_labels: None,
        }
    }

    /// Pin the class-label list that maps probability columns to labels.
    pub fn with_class_labels(mut self, labels: &[i64]) -> Self {
        self.class_labels = Some(labels.to_vec());
        self
    }
}

impl<F: SKFloat, M: SKClassifierPredictor<F>> SKProbabilityScorer<F, M> for SKAveragePrecisionScorer
where
    SKError: From<M::Error>,
{
    type Error = SKError;

    fn score_from_probabilities(
        &self,
        true_targets: SKTargetView<'_, F>,
        probabilities: ArrayView2<'_, F>,
    ) -> Result<F, SKError> {
        let truth = sk_labels_from_targets(&true_targets)?;
        let scores = sk_positive_score_column(
            probabilities,
            self.positive_label,
            self.class_labels.as_deref(),
        )?;
        sk_average_precision_score(
            &scores,
            truth.as_slice().expect("an owned array is contiguous"),
            self.positive_label,
        )
    }
}

/// The log loss of a classifier's class probabilities.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SKLogLossScorer {
    class_labels: Option<Vec<i64>>,
}

impl SKLogLossScorer {
    /// Build a log-loss scorer over the probability columns.
    pub fn new() -> Self {
        Self::default()
    }

    /// Pin the class-label list that maps probability columns to labels.
    pub fn with_class_labels(mut self, labels: &[i64]) -> Self {
        self.class_labels = Some(labels.to_vec());
        self
    }
}

impl<F: SKFloat, M: SKClassifierPredictor<F>> SKProbabilityScorer<F, M> for SKLogLossScorer
where
    SKError: From<M::Error>,
{
    type Error = SKError;

    fn score_from_probabilities(
        &self,
        true_targets: SKTargetView<'_, F>,
        probabilities: ArrayView2<'_, F>,
    ) -> Result<F, SKError> {
        let truth = sk_labels_from_targets(&true_targets)?;
        sk_log_loss(
            truth.as_slice().expect("an owned array is contiguous"),
            probabilities,
            self.class_labels.as_deref(),
        )
    }
}

/// Balanced accuracy over a classifier's labels.
#[derive(Debug, Clone, Copy, Default)]
pub struct SKBalancedAccuracyScorer;

impl<F: SKFloat, M: SKClassifierPredictor<F>> SKLabelScorer<F, M> for SKBalancedAccuracyScorer
where
    SKError: From<M::Error>,
{
    type Error = SKError;

    fn score_from_labels(
        &self,
        true_targets: SKTargetView<'_, F>,
        labels: ArrayView1<'_, i64>,
    ) -> Result<F, SKError> {
        let truth = sk_labels_from_targets(&true_targets)?;
        sk_balanced_accuracy_score(
            truth.as_slice().expect("an owned array is contiguous"),
            labels
                .as_slice()
                .expect("a one-dimensional view is contiguous"),
        )
    }
}

/// The Matthews correlation coefficient over a classifier's labels.
#[derive(Debug, Clone, Copy, Default)]
pub struct SKMatthewsCorrelationScorer;

impl<F: SKFloat, M: SKClassifierPredictor<F>> SKLabelScorer<F, M> for SKMatthewsCorrelationScorer
where
    SKError: From<M::Error>,
{
    type Error = SKError;

    fn score_from_labels(
        &self,
        true_targets: SKTargetView<'_, F>,
        labels: ArrayView1<'_, i64>,
    ) -> Result<F, SKError> {
        let truth = sk_labels_from_targets(&true_targets)?;
        sk_matthews_correlation_coefficient(
            truth.as_slice().expect("an owned array is contiguous"),
            labels
                .as_slice()
                .expect("a one-dimensional view is contiguous"),
        )
    }
}

/// Extract the positive-class score column of a binary probability matrix.
fn sk_positive_score_column<F: SKFloat>(
    probabilities: ArrayView2<'_, F>,
    positive_label: i64,
    class_labels: Option<&[i64]>,
) -> Result<Vec<F>, SKError> {
    if probabilities.ncols() != 2 {
        return Err(SKError::UnsupportedOperation {
            operation: "binary_ranking_scorer",
            reason: "binary ranking metrics require exactly two probability columns".to_string(),
        });
    }
    let column = match class_labels {
        Some(labels) => {
            if labels.len() != probabilities.ncols() {
                return Err(SKError::shape_mismatch_2d(
                    probabilities.ncols(),
                    1,
                    labels.len(),
                    1,
                ));
            }
            let mut sorted = labels.to_vec();
            sorted.sort_unstable();
            sorted
                .binary_search(&positive_label)
                .map_err(|_| SKError::UnknownLabel {
                    label: positive_label,
                })?
        }
        None => match positive_label {
            0 => 0,
            1 => 1,
            _ => {
                return Err(SKError::UnknownLabel {
                    label: positive_label,
                });
            }
        },
    };
    Ok(probabilities.column(column).to_vec())
}
