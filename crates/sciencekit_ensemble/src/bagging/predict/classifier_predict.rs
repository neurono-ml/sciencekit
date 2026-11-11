//! Row-parallel probability averaging and canonical label prediction for
//! [`SKBaggingClassifierModel`].

use std::error::Error;

use ndarray::{Array1, Array2, Axis, Slice};
use sciencekit_common::{
    SKClassifierPredictor, SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind,
    sk_run_operation,
};

use super::super::error::SKBaggingError;
use super::super::fit::sk_map_row_chunks;
use super::super::model::SKBaggingClassifierModel;
use super::super::validation::{
    sk_bagging_predict_dimensions, sk_validate_bagging_finite_features,
};
use super::classifier_routing::sk_predict_classifier_member_chunk;

impl<F, M> SKBaggingClassifierModel<F, M>
where
    F: SKFloat,
    M: SKClassifierPredictor<F> + Send + Sync,
    M::Error: Error + Send + 'static,
{
    /// Predict averaged class probabilities for every row, one column per
    /// canonical class.
    pub fn probabilities<'a, X>(&self, features: X) -> Result<Array2<F>, SKBaggingError<M::Error>>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let data = features.try_into()?;
        let (rows_count, columns_count) = sk_bagging_predict_dimensions(&data, self.feature_count)?;
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Predict,
            rows_count,
            columns_count,
            execution_mode: self.execution_intent,
            backend: self.backend,
        };
        sk_run_operation(attributes, || {
            sk_validate_bagging_finite_features(&data)?;
            let estimator_count =
                F::from(self.members.len()).expect("estimator count fits in scalar");
            let chunks = sk_map_row_chunks(
                rows_count,
                self.members.len(),
                self.execution_intent,
                |row_start, row_end| {
                    let mut sum = Array2::<F>::zeros((row_end - row_start, self.number_of_classes));
                    self.members
                        .iter()
                        .zip(self.feature_subsets.iter())
                        .try_for_each(|(member, feature_subset)| {
                            let part = sk_predict_classifier_member_chunk(
                                member,
                                &data,
                                row_start,
                                row_end,
                                feature_subset,
                                self.number_of_classes,
                            )?;
                            sum.zip_mut_with(&part, |accumulator, &value| {
                                *accumulator = *accumulator + value;
                            });
                            Ok::<(), SKBaggingError<M::Error>>(())
                        })?;
                    sum.mapv_inplace(|value| value / estimator_count);
                    Ok((row_start, row_end, sum))
                },
            )?;

            let mut output = Array2::<F>::zeros((rows_count, self.number_of_classes));
            chunks.into_iter().for_each(|(row_start, row_end, part)| {
                output
                    .slice_axis_mut(Axis(0), Slice::from(row_start..row_end))
                    .assign(&part);
            });
            Ok(output)
        })
    }
}

impl<F, M> SKClassifierPredictor<F> for SKBaggingClassifierModel<F, M>
where
    F: SKFloat,
    M: SKClassifierPredictor<F> + Send + Sync,
    M::Error: Error + Send + 'static,
{
    type Error = SKBaggingError<M::Error>;

    fn predict_labels<'a, X>(&self, features: X) -> Result<Array1<i64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let probabilities = self.probabilities(features)?;
        Ok(Array1::from_shape_fn(probabilities.nrows(), |row| {
            sk_select_canonical_bagging_class(
                probabilities.row(row).as_slice().expect("contiguous row"),
            ) as i64
        }))
    }

    fn predict_probabilities<'a, X>(&self, features: X) -> Result<Array2<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        self.probabilities(features)
    }
}

/// Select the largest class probability, breaking exact ties toward the
/// smallest canonical class index.
pub(in crate::bagging) fn sk_select_canonical_bagging_class<F: SKFloat>(
    probabilities: &[F],
) -> usize {
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
