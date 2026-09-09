//! The unsupervised scorer contract.

use ndarray::Array1;

use crate::SKError;
use crate::data_view::SKDataView;
use crate::fit_traits::SKPredictor;
use crate::sk_float::SKFloat;

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
