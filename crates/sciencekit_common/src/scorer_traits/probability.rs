//! The probability scoring contract.

use ndarray::ArrayView2;

use crate::data_view::SKDataView;
use crate::fit_traits::SKClassifierPredictor;
use crate::sk_float::SKFloat;
use crate::target_view::SKTargetView;

/// A scorer comparing true targets with a classifier's class probabilities.
///
/// The probability form exists alongside the label form because ranking and
/// calibration metrics (ROC/AUC, average precision, log loss) consume the
/// probability matrix, not only arg-max labels. Implementors implement only
/// [`SKProbabilityScorer::score_from_probabilities`]; the convenient
/// [`SKProbabilityScorer::score`] runs `predict_probabilities` and delegates.
/// Scores stay in the model scalar — no `f64` detour.
pub trait SKProbabilityScorer<F: SKFloat, M: SKClassifierPredictor<F>>
where
    Self::Error: From<M::Error>,
{
    /// The evaluation error type; converts from the central [`crate::SKError`].
    type Error: From<crate::SKError>;
    /// Pure form: compare stored class probabilities with true targets — no
    /// inference.
    fn score_from_probabilities(
        &self,
        true_targets: SKTargetView<'_, F>,
        probabilities: ArrayView2<'_, F>,
    ) -> Result<F, Self::Error>;
    /// Convenient form: obtain the model probabilities, then delegate to the
    /// pure form.
    fn score(
        &self,
        model: &M,
        features: SKDataView<'_, F>,
        true_targets: SKTargetView<'_, F>,
    ) -> Result<F, Self::Error> {
        let probabilities = match features {
            SKDataView::Dense(dense) => model.predict_probabilities(dense)?,
            SKDataView::Sparse(sparse) => model.predict_probabilities(sparse)?,
        };
        self.score_from_probabilities(true_targets, probabilities.view())
    }
}
