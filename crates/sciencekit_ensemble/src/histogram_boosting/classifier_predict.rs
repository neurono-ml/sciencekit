//! Probability and label prediction for the histogram classifier.

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKClassifierPredictor, SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind,
    sk_run_operation,
};

use crate::boosting::loss::{SKBinaryLogLoss, SKMulticlassLogLoss};
use crate::boosting::scores::sk_sparse_entry;

use super::binning::SKFeatureBinning;
use super::classifier_model::SKHistGradientBoostingClassifierModel;
use super::validation::{
    sk_map_prediction_rows, sk_predict_dimensions, sk_reject_streaming_mode,
    sk_select_canonical_class, sk_validate_finite_or_missing_features,
};

impl<F: SKFloat> SKHistGradientBoostingClassifierModel<F> {
    /// Predict class probabilities `(rows, C)` through the sigmoid (binary) or
    /// softmax (multiclass) link without densifying sparse inputs.
    pub fn probabilities<'a, X>(&self, features: X) -> Result<Array2<F>, SKError>
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
            let total_trees = self.stages.len().saturating_mul(self.trees_per_stage());
            let rows = match &data {
                SKDataView::Dense(dense) => {
                    sk_map_prediction_rows(rows_count, total_trees, self.execution_intent, |row| {
                        self.evaluate_row_probabilities(|feature| dense[[row, feature]])
                    })
                }
                SKDataView::Sparse(sparse) => {
                    sk_map_prediction_rows(rows_count, total_trees, self.execution_intent, |row| {
                        let sparse_row = sparse.outer_view(row).expect("row within bounds");
                        self.evaluate_row_probabilities(|feature| {
                            sk_sparse_entry(&sparse_row, feature)
                        })
                    })
                }
                _ => unreachable!("dimensions already validated"),
            };
            let flattened: Vec<F> = rows.into_iter().flatten().collect();
            Array2::from_shape_vec((rows_count, self.number_of_classes), flattened)
                .map_err(|error| SKError::Conversion(error.to_string()))
        })
    }

    pub(super) fn evaluate_row_probabilities(&self, feature_value: impl Fn(usize) -> F) -> Vec<F> {
        let bin_of_feature = |feature: usize| {
            sk_bin_feature(&self.feature_binnings[feature], feature_value(feature))
        };
        if self.number_of_classes == 2 {
            let raw_score =
                self.stages
                    .iter()
                    .fold(self.initial_predictions[0], |score, stage_trees| {
                        score + self.learning_rate * stage_trees[0].leaf_value_for(bin_of_feature)
                    });
            let positive = SKBinaryLogLoss::sigmoid(raw_score);
            return vec![F::one() - positive, positive];
        }
        let mut raw_scores = self.initial_predictions.clone();
        self.stages.iter().for_each(|stage_trees| {
            raw_scores
                .iter_mut()
                .zip(stage_trees.iter())
                .for_each(|(score, tree)| {
                    *score = *score + self.learning_rate * tree.leaf_value_for(bin_of_feature);
                });
        });
        SKMulticlassLogLoss::build()
            .number_of_classes(self.number_of_classes)
            .build()
            .expect("fitted class count >= 2")
            .softmax(&raw_scores)
    }
}

/// Bin one raw feature value with its fitted binning; `NaN` follows the
/// missing-category/missing-bin rule.
pub(super) fn sk_bin_feature<F: SKFloat>(binning: &SKFeatureBinning<F>, value: F) -> u8 {
    binning.bin(value)
}

impl<F: SKFloat> SKClassifierPredictor<F> for SKHistGradientBoostingClassifierModel<F> {
    type Error = SKError;

    fn predict_labels<'a, X>(&self, features: X) -> Result<Array1<i64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let probabilities = self.probabilities(features)?;
        Ok(Array1::from_shape_fn(probabilities.nrows(), |row| {
            sk_select_canonical_class(probabilities.row(row).as_slice().expect("contiguous row"))
                as i64
        }))
    }

    fn predict_probabilities<'a, X>(&self, features: X) -> Result<Array2<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        self.probabilities(features)
    }
}
