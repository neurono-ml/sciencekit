//! Additive prediction for the histogram regressor.

use ndarray::Array1;
use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKRegressorPredictor,
    sk_run_operation,
};

use crate::boosting::scores::sk_sparse_entry;

use super::classifier_predict::sk_bin_feature;
use super::regressor_model::SKHistGradientBoostingRegressorModel;
use super::validation::{
    sk_map_prediction_rows, sk_predict_dimensions, sk_reject_streaming_mode,
    sk_validate_finite_or_missing_features,
};

impl<F: SKFloat> SKHistGradientBoostingRegressorModel<F> {
    /// Predict `initial_prediction + Σ learning_rate * stage_tree(x)`.
    pub fn predict_values<'a, X>(&self, features: X) -> Result<Array1<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let data = features.try_into()?;
        let (rows_count, columns_count) = sk_predict_dimensions(&data, self.feature_count)?;
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Predict,
            rows_count,
            columns_count,
            execution_mode: self.execution_intent,
            backend: self.backend,
        };
        sk_run_operation(attributes, || {
            sk_reject_streaming_mode(self.execution_intent)?;
            sk_validate_finite_or_missing_features(&data)?;
            let values = match &data {
                SKDataView::Dense(dense) => sk_map_prediction_rows(
                    rows_count,
                    self.stages.len(),
                    self.execution_intent,
                    |row| self.evaluate_row(|feature| dense[[row, feature]]),
                ),
                SKDataView::Sparse(sparse) => sk_map_prediction_rows(
                    rows_count,
                    self.stages.len(),
                    self.execution_intent,
                    |row| {
                        let sparse_row = sparse.outer_view(row).expect("row within bounds");
                        self.evaluate_row(|feature| sk_sparse_entry(&sparse_row, feature))
                    },
                ),
                _ => unreachable!("dimensions already validated"),
            };
            Ok(Array1::from_vec(values))
        })
    }

    pub(super) fn evaluate_row(&self, feature_value: impl Fn(usize) -> F) -> F {
        let bin_of_feature = |feature: usize| {
            sk_bin_feature(&self.feature_binnings[feature], feature_value(feature))
        };
        self.stages
            .iter()
            .fold(self.initial_prediction, |score, tree| {
                score + self.learning_rate * tree.leaf_value_for(bin_of_feature)
            })
    }
}

impl<F: SKFloat> SKRegressorPredictor<F> for SKHistGradientBoostingRegressorModel<F> {
    type Error = SKError;

    fn predict<'a, X>(&self, features: X) -> Result<Array1<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        self.predict_values(features)
    }
}
