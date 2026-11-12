//! Label-based ranking scorer adapters: balanced accuracy and MCC.

use ndarray::ArrayView1;
use sciencekit_common::{SKClassifierPredictor, SKError, SKFloat, SKLabelScorer, SKTargetView};

use crate::ranking::{sk_balanced_accuracy_score, sk_matthews_correlation_coefficient};
use crate::scorers::targets::sk_labels_from_targets;

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
