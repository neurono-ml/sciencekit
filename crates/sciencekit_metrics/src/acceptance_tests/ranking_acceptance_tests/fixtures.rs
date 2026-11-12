//! Shared fixture data and the model placeholder of the ranking acceptance
//! tests.

use sciencekit_common::{SKClassifierPredictor, SKDataView, SKError, SKFloat};

/// The size of the generated large dataset.
pub(super) const OBSERVATIONS: usize = 100_000;

/// A separable large dataset: the label is the thresholded score.
pub(super) fn large_scores_and_labels() -> (Vec<f64>, Vec<i64>) {
    let scores: Vec<f64> = (0..OBSERVATIONS)
        .map(|index| index as f64 / OBSERVATIONS as f64)
        .collect();
    let labels: Vec<i64> = scores
        .iter()
        .map(|score| if *score >= 0.5 { 1 } else { 0 })
        .collect();
    (scores, labels)
}

/// A model type only needed to name a scorer's generic parameter.
pub(super) struct DummyModel;

impl<F: SKFloat> SKClassifierPredictor<F> for DummyModel {
    type Error = SKError;

    fn predict_labels<'a, X>(&self, _features: X) -> Result<ndarray::Array1<i64>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        Ok(ndarray::Array1::zeros(0))
    }

    fn predict_probabilities<'a, X>(&self, _features: X) -> Result<ndarray::Array2<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        Ok(ndarray::Array2::zeros((0, 0)))
    }
}
