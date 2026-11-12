//! The binary average-precision scorer adapter.

use ndarray::ArrayView2;
use sciencekit_common::{
    SKClassifierPredictor, SKError, SKFloat, SKProbabilityScorer, SKTargetView,
};

use super::column::sk_positive_score_column;
use crate::ranking::sk_average_precision_score;
use crate::scorers::targets::sk_labels_from_targets;

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
