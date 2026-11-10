//! TDD tests for the probability scorer contract (spec `scoring-contracts`).

use ndarray::{Array1, Array2, ArrayView2, array};

use super::SKProbabilityScorer;
use crate::SKError;
use crate::fit_traits::SKClassifierPredictor;
use crate::sk_float::SKFloat;
use crate::{SKDataView, SKTargetView};

/// A classifier that always assigns all mass to the positive class.
struct ConstantProbabilityClassifier;

impl<F: SKFloat> SKClassifierPredictor<F> for ConstantProbabilityClassifier {
    type Error = SKError;

    fn predict_labels<'a, X>(&self, features: X) -> Result<Array1<i64>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view: SKDataView<'_, F> = features.try_into()?;
        let rows = match &view {
            SKDataView::Dense(dense) => dense.nrows(),
            SKDataView::Sparse(sparse) => sparse.rows(),
        };
        Ok(Array1::from_elem(rows, 1_i64))
    }

    fn predict_probabilities<'a, X>(&self, features: X) -> Result<Array2<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let view: SKDataView<'_, F> = features.try_into()?;
        let rows = match &view {
            SKDataView::Dense(dense) => dense.nrows(),
            SKDataView::Sparse(sparse) => sparse.rows(),
        };
        Ok(Array2::from_shape_fn((rows, 2), |(_, column)| {
            if column == 1 { F::one() } else { F::zero() }
        }))
    }
}

/// A classifier whose inference must never run in the pure form.
struct PanickingClassifier;

impl<F: SKFloat> SKClassifierPredictor<F> for PanickingClassifier {
    type Error = SKError;

    fn predict_labels<'a, X>(&self, _features: X) -> Result<Array1<i64>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        panic!("the pure form must not run inference")
    }

    fn predict_probabilities<'a, X>(&self, _features: X) -> Result<Array2<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        panic!("the pure form must not run inference")
    }
}

/// An example probability scorer: the accuracy of the arg-max class.
struct ArgmaxAccuracy;

impl<F: SKFloat, M: SKClassifierPredictor<F>> SKProbabilityScorer<F, M> for ArgmaxAccuracy
where
    SKError: From<M::Error>,
{
    type Error = SKError;

    fn score_from_probabilities(
        &self,
        true_targets: SKTargetView<'_, F>,
        probabilities: ArrayView2<'_, F>,
    ) -> Result<F, SKError> {
        let truth = match &true_targets {
            SKTargetView::Integer(values) => values.to_owned(),
            _ => {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "non-integer",
                    suggestion: "encode targets as integer labels before scoring",
                });
            }
        };
        if truth.len() != probabilities.nrows() {
            return Err(SKError::shape_mismatch_2d(
                truth.len(),
                probabilities.ncols(),
                probabilities.nrows(),
                probabilities.ncols(),
            ));
        }
        let correct = truth
            .iter()
            .zip(probabilities.rows())
            .filter(|(label, row)| {
                let best = row
                    .iter()
                    .enumerate()
                    .max_by(|left, right| {
                        left.1
                            .partial_cmp(right.1)
                            .expect("probabilities are finite")
                    })
                    .map(|(column, _)| column as i64)
                    .unwrap_or(0);
                **label == best
            })
            .count();
        let correct = num_traits::cast::<usize, F>(correct).expect("counts fit in the scalar");
        let total = num_traits::cast::<usize, F>(truth.len()).expect("counts fit in the scalar");
        Ok(correct / total)
    }
}

/// The pure form scores stored probabilities without running inference.
#[test]
fn probability_pure_form_does_not_infer() {
    let truth_labels = array![0_i64, 1, 1];
    let truth = SKTargetView::<f64>::Integer(truth_labels.view());
    let probabilities = array![[0.9_f64, 0.1], [0.2, 0.8], [0.3, 0.7]];
    let score =
        <ArgmaxAccuracy as SKProbabilityScorer<f64, PanickingClassifier>>::score_from_probabilities(
            &ArgmaxAccuracy,
            truth,
            probabilities.view(),
        )
        .expect("valid input");
    assert!((score - 1.0).abs() < 1e-12);
}

/// The constant classifier scores perfectly when every target is its class.
#[test]
fn probability_constant_classifier_scores_perfectly() {
    let model = ConstantProbabilityClassifier;
    let features = Array2::<f64>::zeros((3, 2));
    let truth_labels = array![1_i64, 1, 1];
    let truth = SKTargetView::<f64>::Integer(truth_labels.view());
    let score = ArgmaxAccuracy
        .score(&model, SKDataView::Dense(features.view()), truth)
        .expect("valid input");
    assert!((score - 1.0).abs() < 1e-12);
}

/// The convenient form equals the pure form over the model's probabilities.
#[test]
fn probability_convenient_form_equals_pure_form() {
    let model = ConstantProbabilityClassifier;
    let features = Array2::<f64>::zeros((3, 2));
    let truth_labels = array![1_i64, 0, 1];
    let truth = SKTargetView::<f64>::Integer(truth_labels.view());
    let convenient = ArgmaxAccuracy
        .score(&model, SKDataView::Dense(features.view()), truth)
        .expect("valid input");
    let probabilities = array![[0.0_f64, 1.0], [0.0, 1.0], [0.0, 1.0]];
    let pure =
        <ArgmaxAccuracy as SKProbabilityScorer<f64, ConstantProbabilityClassifier>>::score_from_probabilities(
            &ArgmaxAccuracy,
            truth,
            probabilities.view(),
        )
        .expect("valid input");
    assert!((convenient - pure).abs() < 1e-12);
}
