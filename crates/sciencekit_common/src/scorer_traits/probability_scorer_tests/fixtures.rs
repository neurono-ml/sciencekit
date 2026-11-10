//! Example classifiers and a probability scorer shared by the tests.

use ndarray::{Array1, Array2, ArrayView2};

use super::super::SKProbabilityScorer;
use crate::SKError;
use crate::fit_traits::SKClassifierPredictor;
use crate::sk_float::SKFloat;
use crate::{SKDataView, SKTargetView};

/// A classifier that always assigns all mass to the positive class.
pub(super) struct ConstantProbabilityClassifier;

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
pub(super) struct PanickingClassifier;

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

/// A classifier returning a fixed probability shape.
pub(super) struct WrongShapeClassifier {
    pub(super) rows: usize,
    pub(super) columns: usize,
}

impl<F: SKFloat> SKClassifierPredictor<F> for WrongShapeClassifier {
    type Error = SKError;

    fn predict_labels<'a, X>(&self, _features: X) -> Result<Array1<i64>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        Ok(Array1::zeros(self.rows))
    }

    fn predict_probabilities<'a, X>(&self, _features: X) -> Result<Array2<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        Ok(Array2::zeros((self.rows, self.columns)))
    }
}

/// An example probability scorer: the accuracy of the arg-max class.
pub(super) struct ArgmaxAccuracy;

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
