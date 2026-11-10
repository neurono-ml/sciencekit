//! The probability scoring contract.

use ndarray::ArrayView2;

use crate::data_view::SKDataView;
use crate::fit_traits::SKClassifierPredictor;
use crate::sk_float::SKFloat;
use crate::target_view::SKTargetView;

/// Validate that a probability matrix matches the target count and has at
/// least one class column.
///
/// Empty targets are [`crate::SKError::EmptyInput`]; a row-count disagreement
/// or a matrix without columns is [`crate::SKError::ShapeMismatch`].
pub fn sk_validate_probability_shape<F: SKFloat>(
    number_of_targets: usize,
    probabilities: ArrayView2<'_, F>,
) -> Result<(), crate::SKError> {
    if number_of_targets == 0 {
        return Err(crate::SKError::EmptyInput {
            operation: "probability_scorer",
        });
    }
    if probabilities.nrows() != number_of_targets {
        return Err(crate::SKError::shape_mismatch_2d(
            number_of_targets,
            probabilities.ncols(),
            probabilities.nrows(),
            probabilities.ncols(),
        ));
    }
    if probabilities.ncols() == 0 {
        return Err(crate::SKError::shape_mismatch_2d(
            number_of_targets,
            1,
            number_of_targets,
            0,
        ));
    }
    Ok(())
}

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
        sk_validate_probability_shape(true_targets.len(), probabilities.view())?;
        self.score_from_probabilities(true_targets, probabilities.view())
    }
}
