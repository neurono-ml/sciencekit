//! Regression scorer adapters.

use ndarray::ArrayView1;
use sciencekit_common::{SKError, SKFloat, SKRegressorPredictor, SKSupervisedScorer, SKTargetView};

use crate::regression::{
    sk_mean_absolute_error, sk_mean_squared_error, sk_r2_score, sk_root_mean_squared_error,
};

/// Mean squared error over a regressor's predictions.
#[derive(Debug, Clone, Copy, Default)]
pub struct SKMeanSquaredErrorScorer;

impl<F: SKFloat, M: SKRegressorPredictor<F>> SKSupervisedScorer<F, M> for SKMeanSquaredErrorScorer
where
    SKError: From<M::Error>,
{
    type Error = SKError;

    fn score_from_predictions(
        &self,
        true_targets: SKTargetView<'_, F>,
        predictions: ArrayView1<'_, F>,
    ) -> Result<F, Self::Error> {
        let truth = true_targets.as_continuous()?;
        sk_mean_squared_error(truth.view(), predictions)
    }
}

/// Root mean squared error over a regressor's predictions.
#[derive(Debug, Clone, Copy, Default)]
pub struct SKRootMeanSquaredErrorScorer;

impl<F: SKFloat, M: SKRegressorPredictor<F>> SKSupervisedScorer<F, M>
    for SKRootMeanSquaredErrorScorer
where
    SKError: From<M::Error>,
{
    type Error = SKError;

    fn score_from_predictions(
        &self,
        true_targets: SKTargetView<'_, F>,
        predictions: ArrayView1<'_, F>,
    ) -> Result<F, Self::Error> {
        let truth = true_targets.as_continuous()?;
        sk_root_mean_squared_error(truth.view(), predictions)
    }
}

/// Mean absolute error over a regressor's predictions.
#[derive(Debug, Clone, Copy, Default)]
pub struct SKMeanAbsoluteErrorScorer;

impl<F: SKFloat, M: SKRegressorPredictor<F>> SKSupervisedScorer<F, M> for SKMeanAbsoluteErrorScorer
where
    SKError: From<M::Error>,
{
    type Error = SKError;

    fn score_from_predictions(
        &self,
        true_targets: SKTargetView<'_, F>,
        predictions: ArrayView1<'_, F>,
    ) -> Result<F, Self::Error> {
        let truth = true_targets.as_continuous()?;
        sk_mean_absolute_error(truth.view(), predictions)
    }
}

/// The coefficient of determination over a regressor's predictions.
#[derive(Debug, Clone, Copy, Default)]
pub struct SKR2Scorer;

impl<F: SKFloat, M: SKRegressorPredictor<F>> SKSupervisedScorer<F, M> for SKR2Scorer
where
    SKError: From<M::Error>,
{
    type Error = SKError;

    fn score_from_predictions(
        &self,
        true_targets: SKTargetView<'_, F>,
        predictions: ArrayView1<'_, F>,
    ) -> Result<F, Self::Error> {
        let truth = true_targets.as_continuous()?;
        sk_r2_score(truth.view(), predictions)
    }
}
