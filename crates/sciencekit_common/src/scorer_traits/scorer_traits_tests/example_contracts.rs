//! Example contracts used by the scoring tests (predictors and scorers).
use ndarray::{Array1, Array2, ArrayView1};

use super::super::{SKLabelScorer, SKSupervisedScorer, SKUnsupervisedScorer};
use crate::SKError;
use crate::fit_traits::{SKClassifierPredictor, SKRegressorPredictor};
use crate::{SKDataView, SKTargetView};

// ---- Example contracts ----------------------------------------------------

/// An example regressor: always predicts the response `F::one()`.
pub struct ConstantRegressor;
impl<F: crate::SKFloat> SKRegressorPredictor<F> for ConstantRegressor {
    type Error = SKError;
    fn predict<'a, X>(&self, features: X) -> Result<Array1<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view: SKDataView<'_, F> = features.try_into()?;
        let rows = match &view {
            SKDataView::Dense(dense) => dense.nrows(),
            SKDataView::Sparse(sparse) => sparse.rows(),
        };
        Ok(Array1::from_elem(rows, F::one()))
    }
}

/// Another regressor family: predicts the response `F::zero()`.
pub struct ZeroRegressor;
impl<F: crate::SKFloat> SKRegressorPredictor<F> for ZeroRegressor {
    type Error = SKError;
    fn predict<'a, X>(&self, features: X) -> Result<Array1<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view: SKDataView<'_, F> = features.try_into()?;
        let rows = match &view {
            SKDataView::Dense(dense) => dense.nrows(),
            SKDataView::Sparse(sparse) => sparse.rows(),
        };
        Ok(Array1::from_elem(rows, F::zero()))
    }
}

/// An example classifier: always predicts label index 1.
pub struct ConstantClassifier;
impl<F: crate::SKFloat> SKClassifierPredictor<F> for ConstantClassifier {
    type Error = SKError;
    fn predict_labels<'a, X>(&self, features: X) -> Result<Array1<i64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view: SKDataView<'_, F> = features.try_into()?;
        let rows = match &view {
            SKDataView::Dense(dense) => dense.nrows(),
            SKDataView::Sparse(sparse) => sparse.rows(),
        };
        Ok(Array1::from_elem(rows, 1_i64))
    }
    fn predict_probabilities<'a, X>(&self, features: X) -> Result<Array2<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view: SKDataView<'_, F> = features.try_into()?;
        let rows = match &view {
            SKDataView::Dense(dense) => dense.nrows(),
            SKDataView::Sparse(sparse) => sparse.rows(),
        };
        let mut probabilities = Array2::zeros((rows, 2));
        probabilities.column_mut(1).fill(F::one());
        Ok(probabilities)
    }
}

/// Another classifier family: predicts label index 0.
pub struct ZeroClassifier;
impl<F: crate::SKFloat> SKClassifierPredictor<F> for ZeroClassifier {
    type Error = SKError;
    fn predict_labels<'a, X>(&self, features: X) -> Result<Array1<i64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view: SKDataView<'_, F> = features.try_into()?;
        let rows = match &view {
            SKDataView::Dense(dense) => dense.nrows(),
            SKDataView::Sparse(sparse) => sparse.rows(),
        };
        Ok(Array1::from_elem(rows, 0_i64))
    }
    fn predict_probabilities<'a, X>(&self, features: X) -> Result<Array2<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view: SKDataView<'_, F> = features.try_into()?;
        let rows = match &view {
            SKDataView::Dense(dense) => dense.nrows(),
            SKDataView::Sparse(sparse) => sparse.rows(),
        };
        let mut probabilities = Array2::zeros((rows, 2));
        probabilities.column_mut(0).fill(F::one());
        Ok(probabilities)
    }
}

/// A label accuracy scorer (metric only; provided form is inherited).
pub struct Accuracy;
impl<F: crate::SKFloat, M: SKClassifierPredictor<F, Error = SKError>> SKLabelScorer<F, M>
    for Accuracy
{
    type Error = SKError;
    fn score_from_labels(
        &self,
        true_targets: SKTargetView<'_, F>,
        labels: ArrayView1<'_, i64>,
    ) -> Result<F, Self::Error> {
        let truth = match &true_targets {
            SKTargetView::Integer(integers) => integers.to_owned(),
            SKTargetView::Continuous(responses) => responses
                .mapv(|response| num_traits::ToPrimitive::to_i64(&response.round()).unwrap_or(0)),
            SKTargetView::Nominal(_) => {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "nominal",
                    suggestion: "encode nominal targets to indices before scoring",
                });
            }
        };
        if truth.len() != labels.len() {
            return Err(SKError::shape_mismatch_2d(truth.len(), 1, labels.len(), 1));
        }
        let correct = truth
            .iter()
            .zip(labels.iter())
            .filter(|(truth, label)| truth == label)
            .count();
        let correct = num_traits::cast(correct).unwrap_or(F::zero());
        let total = num_traits::cast(truth.len()).unwrap_or(F::one());
        Ok(correct / total)
    }
}

/// A continuous mean-absolute-error scorer in the model scalar.
pub struct MeanAbsoluteError;
impl<F: crate::SKFloat, M: SKRegressorPredictor<F, Error = SKError>> SKSupervisedScorer<F, M>
    for MeanAbsoluteError
{
    type Error = SKError;
    fn score_from_predictions(
        &self,
        true_targets: SKTargetView<'_, F>,
        predictions: ArrayView1<'_, F>,
    ) -> Result<F, Self::Error> {
        let truth = true_targets.as_continuous()?;
        if truth.len() != predictions.len() {
            return Err(SKError::shape_mismatch_2d(
                truth.len(),
                1,
                predictions.len(),
                1,
            ));
        }
        let total_error = truth
            .iter()
            .zip(predictions.iter())
            .map(|(truth, prediction)| (*truth - *prediction).abs())
            .fold(F::zero(), |accumulated, error| accumulated + error);
        let count = num_traits::cast(truth.len()).unwrap_or(F::one());
        Ok(total_error / count)
    }
}

/// An unsupervised "majority agreement" scorer over assignments.
pub struct MajorityAgreement;
impl<F: crate::SKFloat, M: SKRegressorPredictor<F, Error = SKError>> SKUnsupervisedScorer<F, M>
    for MajorityAgreement
{
    type Error = SKError;
    fn score_from_assignments(
        &self,
        _features: SKDataView<'_, F>,
        assignments: ArrayView1<'_, usize>,
    ) -> Result<f64, Self::Error> {
        let total = assignments.len() as f64;
        let ones = assignments.iter().filter(|&&a| a == 1).count() as f64;
        Ok(ones / total)
    }
}
