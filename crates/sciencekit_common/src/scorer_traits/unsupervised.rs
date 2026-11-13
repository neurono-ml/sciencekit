//! The unsupervised scorer contract.

use ndarray::ArrayView1;

use crate::SKError;
use crate::data_view::SKDataView;
use crate::fit_traits::{SKClusterLabel, SKClustererPredictor, sk_validate_cluster_assignments};
use crate::sk_float::SKFloat;

/// An unsupervised scorer (e.g. silhouette-like) over features and typed
/// cluster assignments.
///
/// The pure form receives assignments the caller already computed; the
/// convenient form obtains them from a fitted clusterer through
/// [`SKClustererPredictor`]. A scorer that meets the noise variant documents
/// whether it excludes or rejects those rows — never treating noise as a
/// cluster.
pub trait SKUnsupervisedScorer<F: SKFloat, M: SKClustererPredictor<F>>
where
    Self::Error: From<M::Error>,
{
    /// The evaluation error type; converts from the central [`SKError`].
    type Error: From<SKError>;
    /// Pure form: score features against already-computed assignments.
    fn score_from_assignments(
        &self,
        features: SKDataView<'_, F>,
        assignments: ArrayView1<'_, SKClusterLabel>,
    ) -> Result<f64, Self::Error>;
    /// Convenient form: obtain the model's typed assignments, then delegate.
    fn score(&self, model: &M, features: SKDataView<'_, F>) -> Result<f64, Self::Error> {
        let (number_of_rows, assignments) = match features {
            SKDataView::Dense(dense) => (dense.nrows(), model.predict_clusters(dense)?),
            SKDataView::Sparse(sparse) => (sparse.rows(), model.predict_clusters(sparse)?),
        };
        sk_validate_cluster_assignments(number_of_rows, assignments.view())?;
        self.score_from_assignments(features, assignments.view())
    }
}
