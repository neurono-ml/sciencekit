//! Row-parallel response averaging for the random-forest regressor.

use ndarray::Array1;
use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKRegressorPredictor,
    sk_run_operation,
};
use sciencekit_tree::SKDecisionTreeRegressorModel;

use super::execution::sk_map_prediction_rows;
use super::input_validation::{
    sk_predict_dimensions, sk_sparse_feature_value, sk_validate_finite_features,
};
use super::regressor_model::SKRandomForestRegressorModel;

impl<F: SKFloat> SKRandomForestRegressorModel<F> {
    /// Predict the mean tree response for every row.
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
            sk_validate_finite_features(&data)?;
            let tree_count = F::from(self.trees.len()).expect("tree count fits in scalar");
            let values = match &data {
                SKDataView::Dense(dense) => sk_map_prediction_rows(
                    rows_count,
                    self.trees.len(),
                    self.execution_intent,
                    |row| {
                        let sum = self.trees.iter().fold(F::zero(), |accumulator, tree| {
                            accumulator
                                + sk_regressor_tree_value(tree, |feature| dense[[row, feature]])
                        });
                        sum / tree_count
                    },
                ),
                SKDataView::Sparse(sparse) => sk_map_prediction_rows(
                    rows_count,
                    self.trees.len(),
                    self.execution_intent,
                    |row| {
                        let sparse_row = sparse.outer_view(row).expect("row within bounds");
                        let sum = self.trees.iter().fold(F::zero(), |accumulator, tree| {
                            accumulator
                                + sk_regressor_tree_value(tree, |feature| {
                                    sk_sparse_feature_value(&sparse_row, feature)
                                })
                        });
                        sum / tree_count
                    },
                ),
                _ => unreachable!("dimensions already validated"),
            };
            Ok(Array1::from_vec(values))
        })
    }
}

impl<F: SKFloat> SKRegressorPredictor<F> for SKRandomForestRegressorModel<F> {
    type Error = SKError;

    fn predict<'a, X>(&self, features: X) -> Result<Array1<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        self.predict_values(features)
    }
}

/// Evaluate one fitted regression tree on a single sample via `feature_value`.
pub(super) fn sk_regressor_tree_value<F: SKFloat>(
    tree: &SKDecisionTreeRegressorModel<F>,
    feature_value: impl Fn(usize) -> F,
) -> F {
    let leaf = tree.tree().leaf_index_for(feature_value);
    tree.tree().node(leaf).expect("leaf exists").value()[0]
}
