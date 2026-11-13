//! Prediction for an attached adapter (spec `lora-adapters`).
//!
//! The adapter never densifies a sparse input and never mutates the base: it
//! folds the low-rank update into the effective coefficient block and applies
//! the same matrix product the base model applies, so a zero adapter and a
//! merged adapter reproduce the base prediction exactly.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sciencekit_common::{
    SKClassifierPredictor, SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind,
    SKRegressorPredictor, sk_run_operation,
};
use sciencekit_math::sk_csr_dense_product;

use super::attached_model::SKLoraAttachedModel;
use super::base_model::SKLoraBaseModel;
use crate::logistic_regression::validation::unsupported_representation;

/// The row and column counts of a feature view.
pub(crate) fn sk_feature_dimensions<F: SKFloat>(data: &SKDataView<'_, F>) -> (usize, usize) {
    match data {
        SKDataView::Dense(dense) => dense.dim(),
        SKDataView::Sparse(sparse) => (sparse.rows(), sparse.cols()),
        _ => (0, 0),
    }
}

/// The operation attributes for one adapter prediction.
pub(crate) fn sk_prediction_attributes<F: SKFloat>(
    attached: &SKLoraAttachedModel<F>,
    rows_count: usize,
    columns_count: usize,
) -> SKOperationAttributes {
    SKOperationAttributes {
        operation: SKOperationKind::Predict,
        rows_count,
        columns_count,
        execution_mode: attached.adapter().execution_mode(),
        backend: attached.adapter().backend(),
    }
}

/// Score one head: `X · coefficients + intercept`.
pub(crate) fn sk_single_head_scores<F: SKFloat>(
    data: &SKDataView<'_, F>,
    coefficients: ArrayView1<'_, F>,
    intercept: F,
) -> Result<Array1<F>, SKError> {
    match data {
        SKDataView::Dense(dense) => Ok(dense.dot(&coefficients).mapv(|value| value + intercept)),
        SKDataView::Sparse(sparse) => {
            let weights = coefficients.insert_axis(Axis(1));
            let product = sk_csr_dense_product(sparse, &weights);
            Ok(product.column(0).mapv(|value| value + intercept))
        }
        _ => Err(unsupported_representation()),
    }
}

/// Score every head: `X · coefficientsᵀ + intercepts`.
pub(crate) fn sk_multi_head_scores<F: SKFloat>(
    data: &SKDataView<'_, F>,
    coefficients: ArrayView2<'_, F>,
    intercepts: ArrayView1<'_, F>,
) -> Result<Array2<F>, SKError> {
    let rows_count = sk_feature_dimensions(data).0;
    let heads = coefficients.nrows();
    let mut scores = Array2::<F>::zeros((rows_count, heads));
    match data {
        SKDataView::Dense(dense) => {
            for head in 0..heads {
                let column = dense.dot(&coefficients.row(head));
                scores
                    .column_mut(head)
                    .assign(&column.mapv(|value| value + intercepts[head]));
            }
        }
        SKDataView::Sparse(sparse) => {
            for head in 0..heads {
                let weights = coefficients.row(head).insert_axis(Axis(1));
                let product = sk_csr_dense_product(sparse, &weights);
                scores
                    .column_mut(head)
                    .assign(&product.column(0).mapv(|value| value + intercepts[head]));
            }
        }
        _ => return Err(unsupported_representation()),
    }
    Ok(scores)
}

impl<F: SKFloat> SKLoraAttachedModel<F> {
    /// The linear decision scores for a logistic base, one column per head.
    ///
    /// # Errors
    ///
    /// Returns [`SKError::UnsupportedOperation`] for a regression base and a
    /// shape error when the feature count disagrees.
    pub fn decision_function<'a, X>(&self, features: X) -> Result<Array2<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        if !matches!(self.base(), SKLoraBaseModel::LogisticRegression(_)) {
            return Err(SKError::UnsupportedOperation {
                operation: "decision_function",
                reason: "only a logistic-regression base exposes decision scores".to_string(),
            });
        }
        let data = features.try_into()?;
        let (rows_count, columns_count) = sk_feature_dimensions(&data);
        if columns_count != self.feature_count() {
            return Err(SKError::shape_mismatch_2d(
                rows_count,
                self.feature_count(),
                rows_count,
                columns_count,
            ));
        }
        let coefficients = self.effective_coefficients()?;
        let intercepts = self.base().intercepts()?;
        let attributes = sk_prediction_attributes(self, rows_count, columns_count);
        sk_run_operation(attributes, || {
            sk_multi_head_scores(&data, coefficients.view(), intercepts.view())
        })
    }
}

impl<F: SKFloat> SKRegressorPredictor<F> for SKLoraAttachedModel<F> {
    type Error = SKError;

    fn predict<'a, X>(&self, features: X) -> Result<Array1<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        if !matches!(self.base(), SKLoraBaseModel::LinearRegression(_)) {
            return Err(SKError::UnsupportedOperation {
                operation: "predict",
                reason: "this adapter targets a classifier; use predict_labels or \
                         predict_probabilities"
                    .to_string(),
            });
        }
        let data = features.try_into()?;
        let (rows_count, columns_count) = sk_feature_dimensions(&data);
        if columns_count != self.feature_count() {
            return Err(SKError::shape_mismatch_2d(
                rows_count,
                self.feature_count(),
                rows_count,
                columns_count,
            ));
        }
        let coefficients = self.effective_coefficients()?;
        let intercept = self.base().intercepts()?[0];
        let attributes = sk_prediction_attributes(self, rows_count, columns_count);
        sk_run_operation(attributes, || {
            sk_single_head_scores(&data, coefficients.row(0), intercept)
        })
    }
}

impl<F: SKFloat> SKClassifierPredictor<F> for SKLoraAttachedModel<F> {
    type Error = SKError;

    fn predict_labels<'a, X>(&self, features: X) -> Result<Array1<i64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let probabilities = self.predict_probabilities(features)?;
        let labels = probabilities.rows().into_iter().map(|row| {
            let mut best = 0usize;
            for (class, value) in row.iter().enumerate() {
                if *value > row[best] {
                    best = class;
                }
            }
            best as i64
        });
        Ok(Array1::from_iter(labels))
    }

    fn predict_probabilities<'a, X>(&self, features: X) -> Result<Array2<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let base = match self.base() {
            SKLoraBaseModel::LogisticRegression(model) => model,
            _ => {
                return Err(SKError::UnsupportedOperation {
                    operation: "predict_labels",
                    reason: "this adapter targets a regressor; use predict".to_string(),
                });
            }
        };
        let scores = self.decision_function(features)?;
        Ok(base.probabilities_from_scores(scores))
    }
}
