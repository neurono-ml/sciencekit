//! Classification scorer adapters.

use ndarray::ArrayView1;
use sciencekit_common::{
    SKClassifierPredictor, SKError, SKFloat, SKLabelScorer, SKTargetView,
};

use super::targets::sk_labels_from_targets;
use crate::classification::{
    SKAveraging, sk_accuracy_score, sk_f1_score, sk_precision_score, sk_recall_score,
};

/// Accuracy over a classifier's labels.
#[derive(Debug, Clone, Copy, Default)]
pub struct SKAccuracyScorer;

impl<F: SKFloat, M: SKClassifierPredictor<F>> SKLabelScorer<F, M> for SKAccuracyScorer
where
    SKError: From<M::Error>,
{
    type Error = SKError;

    fn score_from_labels(
        &self,
        true_targets: SKTargetView<'_, F>,
        labels: ArrayView1<'_, i64>,
    ) -> Result<F, Self::Error> {
        let truth = sk_labels_from_targets(&true_targets)?;
        sk_accuracy_score(
            truth.as_slice().expect("an owned array is contiguous"),
            labels.as_slice().expect("a one-dimensional view is contiguous"),
        )
    }
}

/// Precision over a classifier's labels, with an averaging policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SKPrecisionScorer {
    averaging: SKAveraging,
    labels: Option<Vec<i64>>,
}

impl SKPrecisionScorer {
    /// Build a precision scorer with the chosen averaging policy.
    pub fn new(averaging: SKAveraging) -> Self {
        Self {
            averaging,
            labels: None,
        }
    }

    /// Pin an explicit label space for the score.
    pub fn with_labels(mut self, labels: &[i64]) -> Self {
        self.labels = Some(labels.to_vec());
        self
    }
}

impl<F: SKFloat, M: SKClassifierPredictor<F>> SKLabelScorer<F, M> for SKPrecisionScorer
where
    SKError: From<M::Error>,
{
    type Error = SKError;

    fn score_from_labels(
        &self,
        true_targets: SKTargetView<'_, F>,
        labels: ArrayView1<'_, i64>,
    ) -> Result<F, Self::Error> {
        let truth = sk_labels_from_targets(&true_targets)?;
        sk_precision_score(
            truth.as_slice().expect("an owned array is contiguous"),
            labels.as_slice().expect("a one-dimensional view is contiguous"),
            self.labels.as_deref(),
            self.averaging,
        )
    }
}

/// Recall over a classifier's labels, with an averaging policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SKRecallScorer {
    averaging: SKAveraging,
    labels: Option<Vec<i64>>,
}

impl SKRecallScorer {
    /// Build a recall scorer with the chosen averaging policy.
    pub fn new(averaging: SKAveraging) -> Self {
        Self {
            averaging,
            labels: None,
        }
    }

    /// Pin an explicit label space for the score.
    pub fn with_labels(mut self, labels: &[i64]) -> Self {
        self.labels = Some(labels.to_vec());
        self
    }
}

impl<F: SKFloat, M: SKClassifierPredictor<F>> SKLabelScorer<F, M> for SKRecallScorer
where
    SKError: From<M::Error>,
{
    type Error = SKError;

    fn score_from_labels(
        &self,
        true_targets: SKTargetView<'_, F>,
        labels: ArrayView1<'_, i64>,
    ) -> Result<F, Self::Error> {
        let truth = sk_labels_from_targets(&true_targets)?;
        sk_recall_score(
            truth.as_slice().expect("an owned array is contiguous"),
            labels.as_slice().expect("a one-dimensional view is contiguous"),
            self.labels.as_deref(),
            self.averaging,
        )
    }
}

/// F1 over a classifier's labels, with an averaging policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SKF1Scorer {
    averaging: SKAveraging,
    labels: Option<Vec<i64>>,
}

impl SKF1Scorer {
    /// Build an F1 scorer with the chosen averaging policy.
    pub fn new(averaging: SKAveraging) -> Self {
        Self {
            averaging,
            labels: None,
        }
    }

    /// Pin an explicit label space for the score.
    pub fn with_labels(mut self, labels: &[i64]) -> Self {
        self.labels = Some(labels.to_vec());
        self
    }
}

impl<F: SKFloat, M: SKClassifierPredictor<F>> SKLabelScorer<F, M> for SKF1Scorer
where
    SKError: From<M::Error>,
{
    type Error = SKError;

    fn score_from_labels(
        &self,
        true_targets: SKTargetView<'_, F>,
        labels: ArrayView1<'_, i64>,
    ) -> Result<F, Self::Error> {
        let truth = sk_labels_from_targets(&true_targets)?;
        sk_f1_score(
            truth.as_slice().expect("an owned array is contiguous"),
            labels.as_slice().expect("a one-dimensional view is contiguous"),
            self.labels.as_deref(),
            self.averaging,
        )
    }
}
