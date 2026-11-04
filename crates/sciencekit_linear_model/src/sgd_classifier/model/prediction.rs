//! Prediction for the fitted SGD classifier model (spec
//! `stochastic-gradient-classifier`).

use ndarray::{Array1, Array2, Axis};
use sciencekit_common::{
    SKClassifierPredictor, SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind,
    sk_run_operation,
};
use sciencekit_math::sk_csr_dense_product;

use super::core_implementation::SKSGDClassifierModel;
use crate::sgd::SKSGDLoss;

impl<F: SKFloat> SKSGDClassifierModel<F> {
    /// The raw linear scores, shaped `samples × heads`.
    pub fn decision_function<'a, X>(&self, features: X) -> Result<Array2<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let data = features.try_into()?;
        let (rows_count, columns_count) = dimensions(&data);
        if columns_count != self.feature_count() {
            return Err(self.feature_mismatch(rows_count, columns_count));
        }
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Predict,
            rows_count,
            columns_count,
            execution_mode: self.execution_intent,
            backend: self.backend,
        };
        sk_run_operation(attributes, || {
            let heads = self.number_of_heads();
            let mut scores = Array2::<F>::zeros((rows_count, heads));
            match data {
                SKDataView::Dense(dense) => {
                    for head in 0..heads {
                        let column = dense.dot(&self.coefficients.row(head));
                        scores
                            .column_mut(head)
                            .assign(&column.mapv(|value| value + self.intercepts[head]));
                    }
                }
                SKDataView::Sparse(sparse) => {
                    for head in 0..heads {
                        let weights = self.coefficients.row(head).insert_axis(Axis(1));
                        let product = sk_csr_dense_product(&sparse, &weights.view());
                        scores.column_mut(head).assign(
                            &product
                                .column(0)
                                .mapv(|value| value + self.intercepts[head]),
                        );
                    }
                }
                _ => return Err(unknown_representation()),
            }
            Ok(scores)
        })
    }

    /// The canonical label index of the largest decision per row.
    pub(crate) fn labels_from_scores(&self, scores: &Array2<F>) -> Array1<i64> {
        Array1::from_shape_fn(scores.nrows(), |row| {
            if self.is_binary() {
                if scores[[row, 0]] > F::zero() { 1 } else { 0 }
            } else {
                let mut best = 0usize;
                for head in 1..scores.ncols() {
                    if scores[[row, head]] > scores[[row, best]] {
                        best = head;
                    }
                }
                best as i64
            }
        })
    }

    /// Turn the decision scores into per-class probabilities.
    fn probabilities_from_scores(&self, scores: &Array2<F>) -> Result<Array2<F>, SKError> {
        if !SKSGDLoss::<F>::is_probabilistic(&self.loss) {
            return Err(SKError::UnsupportedOperation {
                operation: "predict_probabilities",
                reason: format!(
                    "the {:?} loss does not produce calibrated probabilities",
                    self.loss
                ),
            });
        }
        let number_of_classes = self.number_of_classes();
        let mut probabilities = Array2::<F>::zeros((scores.nrows(), number_of_classes));
        if self.is_binary() {
            for (mut row, scores_row) in probabilities.outer_iter_mut().zip(scores.outer_iter()) {
                let positive = clamp(SKSGDLoss::<F>::probability(&self.loss, scores_row[0]));
                row[1] = positive;
                row[0] = F::one() - positive;
            }
        } else {
            // Compute the arg-max once for the whole matrix: the zero-total
            // fallback must not re-score every row on every iteration.
            let labels = self.labels_from_scores(scores);
            for ((mut row, scores_row), &best_label) in probabilities
                .outer_iter_mut()
                .zip(scores.outer_iter())
                .zip(labels.iter())
            {
                row.iter_mut()
                    .zip(scores_row.iter())
                    .for_each(|(value, &score)| {
                        *value = clamp(SKSGDLoss::<F>::probability(&self.loss, score));
                    });
                let total = row
                    .iter()
                    .fold(F::zero(), |accumulator, &value| accumulator + value);
                if total > F::zero() {
                    row.iter_mut().for_each(|value| *value = *value / total);
                } else {
                    row[best_label as usize] = F::one();
                }
            }
        }
        Ok(probabilities)
    }
}

impl<F: SKFloat> SKClassifierPredictor<F> for SKSGDClassifierModel<F> {
    type Error = SKError;

    fn predict_labels<'a, X>(&self, features: X) -> Result<Array1<i64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let scores = self.decision_function(features)?;
        Ok(self.labels_from_scores(&scores))
    }

    fn predict_probabilities<'a, X>(&self, features: X) -> Result<Array2<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let scores = self.decision_function(features)?;
        self.probabilities_from_scores(&scores)
    }
}

/// The row and column counts of a feature view.
pub(crate) fn dimensions<F>(data: &SKDataView<'_, F>) -> (usize, usize) {
    match data {
        SKDataView::Dense(dense) => dense.dim(),
        SKDataView::Sparse(sparse) => (sparse.rows(), sparse.cols()),
        _ => (0, 0),
    }
}

/// Clamp a probability to the closed unit interval.
fn clamp<F: SKFloat>(value: F) -> F {
    value.max(F::zero()).min(F::one())
}

/// The error for a representation prediction cannot consume.
fn unknown_representation() -> SKError {
    SKError::UnsupportedRepresentation {
        representation: "unknown",
        suggestion: "convert the features to a dense matrix or a CSR matrix",
    }
}
