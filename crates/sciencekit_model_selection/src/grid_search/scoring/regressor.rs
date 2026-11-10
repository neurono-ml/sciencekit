//! The regressor scoring policy and its metric dispatch.
//!
//! Every policy is "higher is better": the error metrics are negated, mirroring
//! the reference semantics of negative mean squared error and negative mean
//! absolute error, so candidate selection always maximises the score.

use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKRegressorPredictor, SKSupervisedScorer, SKTargetView,
};
use sciencekit_metrics::scorers::{
    SKMeanAbsoluteErrorScorer, SKMeanSquaredErrorScorer, SKR2Scorer,
};

use super::SKScoringFamily;
use crate::grid_search::evaluation::SKFoldScorer;

/// The closed set of regression scoring policies.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SKRegressorScoring {
    /// The negative mean squared error.
    #[default]
    NegativeMeanSquaredError,
    /// The negative mean absolute error.
    NegativeMeanAbsoluteError,
    /// The coefficient of determination (`R²`).
    R2,
}

impl SKRegressorScoring {
    /// The estimator family this policy drives.
    pub fn family(&self) -> SKScoringFamily {
        SKScoringFamily::Regressor
    }

    fn score_model<F, M>(
        &self,
        model: &M,
        features: SKDataView<'_, F>,
        targets: SKTargetView<'_, F>,
    ) -> Result<F, SKError>
    where
        F: SKFloat,
        M: SKRegressorPredictor<F>,
        SKError: From<M::Error>,
    {
        let raw = match self {
            SKRegressorScoring::NegativeMeanSquaredError => {
                SKMeanSquaredErrorScorer.score(model, features, targets)
            }
            SKRegressorScoring::NegativeMeanAbsoluteError => {
                SKMeanAbsoluteErrorScorer.score(model, features, targets)
            }
            SKRegressorScoring::R2 => SKR2Scorer.score(model, features, targets),
        }?;
        Ok(match self {
            SKRegressorScoring::R2 => raw,
            SKRegressorScoring::NegativeMeanSquaredError
            | SKRegressorScoring::NegativeMeanAbsoluteError => F::zero() - raw,
        })
    }
}

impl<F, M> SKFoldScorer<F, M> for SKRegressorScoring
where
    F: SKFloat,
    M: SKRegressorPredictor<F>,
    SKError: From<M::Error>,
{
    fn score(
        &self,
        model: &M,
        features: SKDataView<'_, F>,
        targets: SKTargetView<'_, F>,
    ) -> Result<F, SKError> {
        self.score_model(model, features, targets)
    }
}
