//! The log-loss scorer adapter.

use ndarray::ArrayView2;
use sciencekit_common::{
    SKClassifierPredictor, SKError, SKFloat, SKProbabilityScorer, SKTargetView,
};

use crate::ranking::sk_log_loss;
use crate::scorers::targets::sk_labels_from_targets;

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
