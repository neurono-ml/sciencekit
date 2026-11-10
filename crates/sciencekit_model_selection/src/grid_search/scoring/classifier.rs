//! The classifier scoring policy and its metric dispatch.

use sciencekit_common::{
    SKClassifierPredictor, SKDataView, SKError, SKFloat, SKLabelScorer, SKProbabilityScorer,
    SKTargetView,
};
use sciencekit_metrics::classification::SKAveraging;
use sciencekit_metrics::scorers::{
    SKAccuracyScorer, SKAucScorer, SKBalancedAccuracyScorer, SKF1Scorer, SKLogLossScorer,
    SKPrecisionScorer, SKRecallScorer,
};

use super::SKScoringFamily;
use crate::grid_search::evaluation::SKFoldScorer;

/// The closed set of classification scoring policies.
///
/// Every policy is expressed through the shared scoring contracts of
/// `sciencekit_common`, so a custom metric can reuse those contracts directly;
/// the enum only ships the standard policies.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SKClassifierScoring {
    /// Fraction of correctly predicted labels.
    #[default]
    Accuracy,
    /// Mean per-class recall, insensitive to class imbalance.
    BalancedAccuracy,
    /// F1 with the chosen averaging policy.
    F1(SKAveraging),
    /// Precision with the chosen averaging policy.
    Precision(SKAveraging),
    /// Recall with the chosen averaging policy.
    Recall(SKAveraging),
    /// Binary ROC AUC over the probability column of the positive label.
    Auc {
        /// The positive label whose probability column ranks the rows.
        positive_label: i64,
    },
    /// Logarithmic loss over the class probabilities (lower is better).
    LogLoss,
}

impl SKClassifierScoring {
    /// The estimator family this policy drives.
    pub fn family(&self) -> SKScoringFamily {
        SKScoringFamily::Classifier
    }

    /// Whether the policy consumes class probabilities instead of labels.
    pub fn requires_probabilities(&self) -> bool {
        matches!(
            self,
            SKClassifierScoring::Auc { .. } | SKClassifierScoring::LogLoss
        )
    }

    fn sk_score_model<F, M>(
        &self,
        model: &M,
        features: SKDataView<'_, F>,
        targets: SKTargetView<'_, F>,
    ) -> Result<F, SKError>
    where
        F: SKFloat,
        M: SKClassifierPredictor<F>,
        SKError: From<M::Error>,
    {
        match self {
            SKClassifierScoring::Accuracy => {
                SKAccuracyScorer.score(model, features, targets)
            }
            SKClassifierScoring::BalancedAccuracy => SKBalancedAccuracyScorer
                .score(model, features, targets),
            SKClassifierScoring::F1(averaging) => SKF1Scorer::new(*averaging)
                .score(model, features, targets),
            SKClassifierScoring::Precision(averaging) => SKPrecisionScorer::new(*averaging)
                .score(model, features, targets),
            SKClassifierScoring::Recall(averaging) => SKRecallScorer::new(*averaging)
                .score(model, features, targets),
            SKClassifierScoring::Auc { positive_label } => {
                SKAucScorer::new(*positive_label)
                    .score(model, features, targets)
            }
            SKClassifierScoring::LogLoss => SKLogLossScorer::new()
                .score(model, features, targets),
        }
    }
}

impl<F, M> SKFoldScorer<F, M> for SKClassifierScoring
where
    F: SKFloat,
    M: SKClassifierPredictor<F>,
    SKError: From<M::Error>,
{
    fn sk_score(
        &self,
        model: &M,
        features: SKDataView<'_, F>,
        targets: SKTargetView<'_, F>,
    ) -> Result<F, SKError> {
        self.sk_score_model(model, features, targets)
    }
}
