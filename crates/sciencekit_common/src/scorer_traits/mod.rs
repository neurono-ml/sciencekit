//! Evaluation contracts (spec `scoring-contracts`).
//!
//! Supervised and unsupervised scorers with **dual input**: a pure form over
//! already-existing predictions/assignments (no re-inference) and a convenient
//! provided form that runs inference and delegates. Both are fallible. Scorers
//! are generic over the evaluated model so one scorer serves many algorithm
//! families. Defined here because they are part of the stable public contract;
//! concrete metrics arrive with the `sciencekit_metrics` crate.

use ndarray::Array1;

use crate::SKError;
use crate::data_view::SKDataView;
use crate::fit_traits::SKPredictor;
use crate::sk_float::SKFloat;
use crate::target_view::SKTargetView;

/// A supervised scorer comparing true targets with predictions.
///
/// Scorers are generic over the model `M` so the same scorer evaluates models
/// from distinct families. Implementors implement only [`SKSupervisedScorer::score_from_predictions`];
/// the convenient [`SKSupervisedScorer::score`] runs inference and delegates.
pub trait SKSupervisedScorer<F: SKFloat, M: SKPredictor<F>>
where
    Self::Error: From<M::Error>,
{
    /// The evaluation error type; converts from the central [`SKError`].
    type Error: From<SKError>;
    /// Pure form: compare stored predictions with true targets — no inference.
    fn score_from_predictions(
        &self,
        true_targets: SKTargetView<'_>,
        predictions: ndarray::ArrayView1<'_, f64>,
    ) -> Result<f64, Self::Error>;
    /// Convenient form: infer from the model, then delegate to the pure form.
    fn score(
        &self,
        model: &M,
        features: SKDataView<'_, F>,
        true_targets: SKTargetView<'_>,
    ) -> Result<f64, Self::Error> {
        // The seam accepts native representations; destructure the view so the
        // underlying representation flows through `TryInto` into `predict`.
        let predictions = match features {
            SKDataView::Dense(d) => model.predict(d)?,
            SKDataView::Sparse(s) => model.predict(s)?,
        };
        self.score_from_predictions(true_targets, predictions.view())
    }
}

/// An unsupervised scorer (e.g. silhouette-like) over features and assignments.
pub trait SKUnsupervisedScorer<F: SKFloat, M: SKPredictor<F>>
where
    Self::Error: From<M::Error>,
{
    /// The evaluation error type; converts from the central [`SKError`].
    type Error: From<SKError>;
    /// Pure form: score features against already-computed assignments.
    fn score_from_assignments(
        &self,
        features: SKDataView<'_, F>,
        assignments: ndarray::ArrayView1<'_, usize>,
    ) -> Result<f64, Self::Error>;
    /// Convenient form: obtain the model's outputs, then delegate to the pure form.
    fn score(&self, model: &M, features: SKDataView<'_, F>) -> Result<f64, Self::Error> {
        let raw = match features {
            SKDataView::Dense(d) => model.predict(d)?,
            SKDataView::Sparse(s) => model.predict(s)?,
        };
        // A predictor returns continuous scores; reinterpret as assignments by
        // rounding to indices (convenient-form contract).
        let assignments: Array1<usize> = raw.mapv(|v| v.round().max(0.0) as usize);
        self.score_from_assignments(features, assignments.view())
    }
}

#[cfg(test)]
mod scorer_traits_tests;
