//! The unsupervised scorer contract.

use ndarray::Array1;

use crate::SKError;
use crate::data_view::SKDataView;
use crate::fit_traits::SKRegressorPredictor;
use crate::sk_float::SKFloat;

/// An unsupervised scorer (e.g. silhouette-like) over features and assignments.
pub trait SKUnsupervisedScorer<F: SKFloat, M: SKRegressorPredictor<F>>
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
            SKDataView::Dense(dense) => model.predict(dense)?,
            SKDataView::Sparse(sparse) => model.predict(sparse)?,
        };
        // A regressor returns continuous responses; reinterpret as assignments
        // by rounding to indices (convenient-form contract).
        let assignments: Array1<usize> =
            raw.mapv(|value| num_traits::ToPrimitive::to_usize(&value.round()).unwrap_or(0));
        self.score_from_assignments(features, assignments.view())
    }
}
