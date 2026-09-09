//! The supervised scorer contract.

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
