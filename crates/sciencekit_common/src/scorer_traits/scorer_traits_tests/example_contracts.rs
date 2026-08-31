//! Example contracts used by the scoring tests (predictors and scorers).
use ndarray::{Array1, ArrayView1};

use super::super::{SKSupervisedScorer, SKUnsupervisedScorer};
use crate::SKError;
use crate::fit_traits::SKPredictor;
use crate::{SKDataView, SKTargetView};

// ---- Example contracts ----------------------------------------------------

/// An example predictor: always predicts label index 1.
pub struct ConstantPredictor;
impl<F: crate::SKFloat> SKPredictor<F> for ConstantPredictor {
    type Error = SKError;
    fn predict<'a, X>(&self, features: X) -> Result<Array1<f64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view: SKDataView<'_, F> = features.try_into()?;
        let rows = match &view {
            SKDataView::Dense(d) => d.nrows(),
            SKDataView::Sparse(s) => s.rows(),
        };
        Ok(Array1::from_elem(rows, 1.0))
    }
}

/// Another predictor family: predicts label index 0.
pub struct ZeroPredictor;
impl<F: crate::SKFloat> SKPredictor<F> for ZeroPredictor {
    type Error = SKError;
    fn predict<'a, X>(&self, features: X) -> Result<Array1<f64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view: SKDataView<'_, F> = features.try_into()?;
        let rows = match &view {
            SKDataView::Dense(d) => d.nrows(),
            SKDataView::Sparse(s) => s.rows(),
        };
        Ok(Array1::from_elem(rows, 0.0))
    }
}

/// A supervised accuracy scorer (metric only; provided form is inherited).
pub struct Accuracy;
impl<F: crate::SKFloat, M: SKPredictor<F, Error = SKError>> SKSupervisedScorer<F, M> for Accuracy {
    type Error = SKError;
    fn score_from_predictions(
        &self,
        true_targets: SKTargetView<'_>,
        predictions: ArrayView1<'_, f64>,
    ) -> Result<f64, Self::Error> {
        let truth = true_targets.as_continuous()?;
        if truth.len() != predictions.len() {
            return Err(SKError::shape_mismatch_2d(
                truth.len(),
                1,
                predictions.len(),
                1,
            ));
        }
        let correct = truth
            .iter()
            .zip(predictions.iter())
            .filter(|(t, p)| t.round() == p.round())
            .count();
        Ok(correct as f64 / truth.len() as f64)
    }
}

/// An unsupervised "majority agreement" scorer over assignments.
pub struct MajorityAgreement;
impl<F: crate::SKFloat, M: SKPredictor<F, Error = SKError>> SKUnsupervisedScorer<F, M>
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
