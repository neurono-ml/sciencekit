//! Supervised scoring contracts: continuous and label scorers.

use crate::SKError;
use crate::data_view::SKDataView;
use crate::fit_traits::{SKClassifierPredictor, SKRegressorPredictor};
use crate::sk_float::SKFloat;
use crate::target_view::SKTargetView;

/// A continuous scorer comparing true targets with regressor predictions.
///
/// Scorers are generic over the model `M` so the same scorer evaluates models
/// from distinct families. Implementors implement only
/// [`SKSupervisedScorer::score_from_predictions`]; the convenient
/// [`SKSupervisedScorer::score`] runs inference and delegates. Scores stay in
/// the model scalar — no `f64` detour.
pub trait SKSupervisedScorer<F: SKFloat, M: SKRegressorPredictor<F>>
where
    Self::Error: From<M::Error>,
{
    /// The evaluation error type; converts from the central [`SKError`].
    type Error: From<SKError>;
    /// Pure form: compare stored predictions with true targets — no inference.
    fn score_from_predictions(
        &self,
        true_targets: SKTargetView<'_, F>,
        predictions: ndarray::ArrayView1<'_, F>,
    ) -> Result<F, Self::Error>;
    /// Convenient form: infer from the model, then delegate to the pure form.
    fn score(
        &self,
        model: &M,
        features: SKDataView<'_, F>,
        true_targets: SKTargetView<'_, F>,
    ) -> Result<F, Self::Error> {
        // The seam accepts native representations; destructure the view so the
        // underlying representation flows through `TryInto` into `predict`.
        let predictions = match features {
            SKDataView::Dense(dense) => model.predict(dense)?,
            SKDataView::Sparse(sparse) => model.predict(sparse)?,
        };
        self.score_from_predictions(true_targets, predictions.view())
    }
}

/// A label scorer comparing true targets with classifier labels.
///
/// Labels are canonical `i64` indices independent of the feature scalar;
/// scores stay in the model scalar. Implementors implement only
/// [`SKLabelScorer::score_from_labels`]; the convenient [`SKLabelScorer::score`]
/// runs inference and delegates.
pub trait SKLabelScorer<F: SKFloat, M: SKClassifierPredictor<F>>
where
    Self::Error: From<M::Error>,
{
    /// The evaluation error type; converts from the central [`SKError`].
    type Error: From<SKError>;
    /// Pure form: compare stored labels with true targets — no inference.
    fn score_from_labels(
        &self,
        true_targets: SKTargetView<'_, F>,
        labels: ndarray::ArrayView1<'_, i64>,
    ) -> Result<F, Self::Error>;
    /// Convenient form: infer labels from the model, then delegate.
    fn score(
        &self,
        model: &M,
        features: SKDataView<'_, F>,
        true_targets: SKTargetView<'_, F>,
    ) -> Result<F, Self::Error> {
        let labels = match features {
            SKDataView::Dense(dense) => model.predict_labels(dense)?,
            SKDataView::Sparse(sparse) => model.predict_labels(sparse)?,
        };
        self.score_from_labels(true_targets, labels.view())
    }
}
