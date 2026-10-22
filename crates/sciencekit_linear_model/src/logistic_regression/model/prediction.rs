//! Prediction for the fitted logistic-regression model (spec
//! `logistic-regression`).

use ndarray::{Array1, Array2, Axis};
use sciencekit_common::{
    SKClassifierPredictor, SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind,
    sk_run_operation,
};
use sciencekit_math::sk_csr_dense_product;

use super::SKLogisticRegressionModel;
use crate::logistic::objective::{sk_sigmoid, sk_softmax};
use crate::logistic::resolution::SKResolvedMultiClass;
use crate::logistic_regression::core_implementation::sk_dimensions;

impl<F: SKFloat> SKLogisticRegressionModel<F> {
    /// The linear decision function, one column per head.
    pub fn decision_function<'a, X>(&self, features: X) -> Result<Array2<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let data = features.try_into()?;
        let (rows_count, columns_count) = sk_dimensions(&data);
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
            let heads = self.coefficients.nrows();
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
                _ => return Err(self.unknown_representation()),
            }
            Ok(scores)
        })
    }

    /// Turn the decision scores into class probabilities.
    pub(crate) fn probabilities_from_scores(&self, scores: Array2<F>) -> Array2<F> {
        let rows_count = scores.nrows();
        let mut probabilities = Array2::<F>::zeros((rows_count, self.number_of_classes));
        match self.strategy {
            SKResolvedMultiClass::Binary => {
                for row in 0..rows_count {
                    let positive = sk_sigmoid(scores[[row, 0]]);
                    probabilities[[row, 1]] = positive;
                    probabilities[[row, 0]] = F::one() - positive;
                }
            }
            SKResolvedMultiClass::Multinomial => {
                let mut logits = vec![F::zero(); self.number_of_classes];
                for row in 0..rows_count {
                    logits.copy_from_slice(scores.row(row).as_slice().expect("row is contiguous"));
                    let mut row_probabilities = vec![F::zero(); self.number_of_classes];
                    sk_softmax(&logits, &mut row_probabilities);
                    for class in 0..self.number_of_classes {
                        probabilities[[row, class]] = row_probabilities[class];
                    }
                }
            }
            SKResolvedMultiClass::OneVersusRest => {
                for row in 0..rows_count {
                    let mut total = F::zero();
                    for class in 0..self.number_of_classes {
                        let value = sk_sigmoid(scores[[row, class]]);
                        probabilities[[row, class]] = value;
                        total = total + value;
                    }
                    let inverse = F::one() / total;
                    for class in 0..self.number_of_classes {
                        probabilities[[row, class]] = probabilities[[row, class]] * inverse;
                    }
                }
            }
        }
        probabilities
    }

    /// The canonical label index with the largest probability, per row.
    fn labels_from_probabilities(&self, probabilities: &Array2<F>) -> Array1<i64> {
        Array1::from_shape_fn(probabilities.nrows(), |row| {
            let mut best = 0usize;
            for class in 1..self.number_of_classes {
                if probabilities[[row, class]] > probabilities[[row, best]] {
                    best = class;
                }
            }
            best as i64
        })
    }

    /// The error for a feature representation prediction cannot consume.
    fn unknown_representation(&self) -> SKError {
        SKError::UnsupportedRepresentation {
            representation: "unknown",
            suggestion: "convert the features to a dense matrix or a CSR matrix",
        }
    }
}

impl<F: SKFloat> SKClassifierPredictor<F> for SKLogisticRegressionModel<F> {
    type Error = SKError;

    /// Predict canonical class indices.
    fn predict_labels<'a, X>(&self, features: X) -> Result<Array1<i64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let probabilities = self.predict_probabilities(features)?;
        Ok(self.labels_from_probabilities(&probabilities))
    }

    /// Predict class probabilities, one column per class.
    fn predict_probabilities<'a, X>(&self, features: X) -> Result<Array2<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let scores = self.decision_function(features)?;
        Ok(self.probabilities_from_scores(scores))
    }
}
