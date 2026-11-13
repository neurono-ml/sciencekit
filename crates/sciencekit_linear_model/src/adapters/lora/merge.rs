//! Merge and unmerge of an attached adapter (spec `lora-adapters`, decision 3).
//!
//! `merge` folds `W + (alpha / rank) · B A` into a standalone native model;
//! `unmerge` restores the frozen base and hands the adapter back. Prediction is
//! shared with the attached model, so merge parity is exact, not approximate.

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKClassifierPredictor, SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind,
    SKRegressorPredictor, sk_run_operation,
};

use super::attached_model::SKLoraAttachedModel;
use super::base_model::SKLoraBaseModel;
use super::core_implementation::SKLoraAdapter;
use super::prediction::{sk_feature_dimensions, sk_multi_head_scores, sk_single_head_scores};

/// A native model with the adapter's low-rank update folded into its
/// coefficients.
///
/// The struct keeps the original frozen base and the factors so [`Self::unmerge`]
/// restores exactly what was merged. The merged coefficients are a real block
/// of the native model returned by [`Self::model`].
#[derive(Debug, Clone)]
pub struct SKLoraMergedModel<F> {
    model: SKLoraBaseModel<F>,
    original: SKLoraBaseModel<F>,
    adapter: SKLoraAdapter<F>,
    a_matrix: Array2<F>,
    b_matrix: Array2<F>,
    fitted: bool,
}

impl<F: SKFloat> SKLoraMergedModel<F> {
    /// The standalone native model with the merged coefficient block.
    pub fn model(&self) -> &SKLoraBaseModel<F> {
        &self.model
    }

    /// The adapter configuration that was merged.
    pub fn adapter(&self) -> &SKLoraAdapter<F> {
        &self.adapter
    }

    /// Whether the adapter had been fitted when it was merged.
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }

    /// Restore the frozen base and the attached adapter.
    pub fn unmerge(self) -> SKLoraAttachedModel<F> {
        SKLoraAttachedModel::from_factors(
            self.adapter,
            self.original,
            self.a_matrix,
            self.b_matrix,
            self.fitted,
        )
    }

    /// The linear decision scores for a logistic base, one column per head.
    pub fn decision_function<'a, X>(&self, features: X) -> Result<Array2<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let base = match &self.model {
            SKLoraBaseModel::LogisticRegression(model) => model,
            _ => {
                return Err(SKError::UnsupportedOperation {
                    operation: "decision_function",
                    reason: "only a logistic-regression base exposes decision scores".to_string(),
                });
            }
        };
        let data = features.try_into()?;
        let (rows_count, columns_count) = sk_feature_dimensions(&data);
        if columns_count != base.feature_count() {
            return Err(SKError::shape_mismatch_2d(
                rows_count,
                base.feature_count(),
                rows_count,
                columns_count,
            ));
        }
        let coefficients = base.coefficients();
        let intercepts = base.intercepts();
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Predict,
            rows_count,
            columns_count,
            execution_mode: self.adapter.execution_mode(),
            backend: self.adapter.backend(),
        };
        sk_run_operation(attributes, || {
            sk_multi_head_scores(&data, coefficients, intercepts)
        })
    }
}

impl<F: SKFloat> SKRegressorPredictor<F> for SKLoraMergedModel<F> {
    type Error = SKError;

    fn predict<'a, X>(&self, features: X) -> Result<Array1<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let base = match &self.model {
            SKLoraBaseModel::LinearRegression(model) => model,
            _ => {
                return Err(SKError::UnsupportedOperation {
                    operation: "predict",
                    reason: "this merged model targets a classifier; use predict_labels"
                        .to_string(),
                });
            }
        };
        let data = features.try_into()?;
        let (rows_count, columns_count) = sk_feature_dimensions(&data);
        if columns_count != base.coefficients().len() {
            return Err(SKError::shape_mismatch_2d(
                rows_count,
                base.coefficients().len(),
                rows_count,
                columns_count,
            ));
        }
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Predict,
            rows_count,
            columns_count,
            execution_mode: self.adapter.execution_mode(),
            backend: self.adapter.backend(),
        };
        let coefficients = base.coefficients();
        let intercept = base.intercept();
        sk_run_operation(attributes, || {
            sk_single_head_scores(&data, coefficients.view(), intercept)
        })
    }
}

impl<F: SKFloat> SKClassifierPredictor<F> for SKLoraMergedModel<F> {
    type Error = SKError;

    fn predict_labels<'a, X>(&self, features: X) -> Result<Array1<i64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let probabilities = self.predict_probabilities(features)?;
        Ok(Array1::from_iter(probabilities.rows().into_iter().map(
            |row| {
                let mut best = 0usize;
                for (class, value) in row.iter().enumerate() {
                    if *value > row[best] {
                        best = class;
                    }
                }
                best as i64
            },
        )))
    }

    fn predict_probabilities<'a, X>(&self, features: X) -> Result<Array2<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let base = match &self.model {
            SKLoraBaseModel::LogisticRegression(model) => model,
            _ => {
                return Err(SKError::UnsupportedOperation {
                    operation: "predict_labels",
                    reason: "this merged model targets a regressor; use predict".to_string(),
                });
            }
        };
        let scores = self.decision_function(features)?;
        Ok(base.probabilities_from_scores(scores))
    }
}

impl<F: SKFloat> SKLoraAttachedModel<F> {
    /// Fold the adapter into a standalone native model.
    ///
    /// # Errors
    ///
    /// Returns shape or unsupported-family errors when the base cannot expose
    /// its coefficient block.
    pub fn merge(&self) -> Result<SKLoraMergedModel<F>, SKError> {
        let effective = self.effective_coefficients()?;
        let merged_base = match self.base() {
            SKLoraBaseModel::LinearRegression(model) => SKLoraBaseModel::LinearRegression(
                model.with_coefficients(effective.row(0).to_owned()),
            ),
            SKLoraBaseModel::LogisticRegression(model) => {
                SKLoraBaseModel::LogisticRegression(model.with_coefficients(effective))
            }
            SKLoraBaseModel::UnsupportedFamily { .. } => {
                return Err(self.base().unsupported_error());
            }
        };
        Ok(SKLoraMergedModel {
            model: merged_base,
            original: self.base().clone(),
            adapter: self.adapter().clone(),
            a_matrix: self.a_matrix.clone(),
            b_matrix: self.b_matrix.clone(),
            fitted: self.is_fitted(),
        })
    }
}
