//! Row-parallel probability averaging and label prediction for the
//! random-forest classifier.

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKClassifierPredictor, SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind,
    sk_run_operation,
};

use super::classifier_model::{SKClassifierTreeMember, SKRandomForestClassifierModel};
use super::execution::sk_map_prediction_rows;
use super::input_validation::{
    sk_predict_dimensions, sk_sparse_feature_value, sk_validate_finite_features,
};

impl<F: SKFloat> SKRandomForestClassifierModel<F> {
    /// Predict averaged class probabilities for every row, one column per
    /// canonical class.
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
            sk_validate_finite_features(&data)?;
            let tree_count = F::from(self.members.len()).expect("tree count fits in scalar");
            let rows = match &data {
                SKDataView::Dense(dense) => sk_map_prediction_rows(
                    rows_count,
                    self.members.len(),
                    self.execution_intent,
                    |row| {
                        let mut distribution = vec![F::zero(); self.number_of_classes];
                        self.members.iter().for_each(|member| {
                            sk_accumulate_member_probabilities(
                                member,
                                &mut distribution,
                                |feature| dense[[row, feature]],
                            );
                        });
                        distribution
                            .iter_mut()
                            .for_each(|value| *value = *value / tree_count);
                        distribution
                    },
                ),
                SKDataView::Sparse(sparse) => sk_map_prediction_rows(
                    rows_count,
                    self.members.len(),
                    self.execution_intent,
                    |row| {
                        let sparse_row = sparse.outer_view(row).expect("row within bounds");
                        let mut distribution = vec![F::zero(); self.number_of_classes];
                        self.members.iter().for_each(|member| {
                            sk_accumulate_member_probabilities(
                                member,
                                &mut distribution,
                                |feature| sk_sparse_feature_value(&sparse_row, feature),
                            );
                        });
                        distribution
                            .iter_mut()
                            .for_each(|value| *value = *value / tree_count);
                        distribution
                    },
                ),
                _ => unreachable!("dimensions already validated"),
            };

            let flattened: Vec<F> = rows.into_iter().flatten().collect();
            Array2::from_shape_vec((rows_count, self.number_of_classes), flattened)
                .map_err(|error| SKError::Conversion(error.to_string()))
        })
    }
}

impl<F: SKFloat> SKClassifierPredictor<F> for SKRandomForestClassifierModel<F> {
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

/// Add one tree member's leaf distribution into `target` in canonical class
/// order.
pub(super) fn sk_accumulate_member_probabilities<F: SKFloat>(
    member: &SKClassifierTreeMember<F>,
    target: &mut [F],
    feature_value: impl Fn(usize) -> F,
) {
    match member {
        SKClassifierTreeMember::Fitted {
            model,
            local_to_forest,
        } => {
            let leaf = model.tree().leaf_index_for(feature_value);
            let values = model.tree().node(leaf).expect("leaf exists").value();
            local_to_forest
                .iter()
                .zip(values.iter())
                .for_each(|(&forest_class, &probability)| {
                    target[forest_class] = target[forest_class] + probability;
                });
        }
        SKClassifierTreeMember::PureLeaf { class_index } => {
            target[*class_index] = target[*class_index] + F::one();
        }
    }
}

/// Select the largest class probability, breaking exact ties toward the
/// smallest canonical class index.
pub(super) fn sk_select_canonical_class<F: SKFloat>(probabilities: &[F]) -> usize {
    probabilities
        .iter()
        .enumerate()
        .skip(1)
        .fold(0usize, |best, (class_index, &probability)| {
            if probability > probabilities[best] {
                class_index
            } else {
                best
            }
        })
}
