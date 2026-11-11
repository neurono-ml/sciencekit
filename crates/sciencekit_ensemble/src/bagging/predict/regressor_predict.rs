//! Row-parallel response averaging for [`SKBaggingRegressorModel`].

use std::error::Error;

use ndarray::{Array1, Axis, Slice};
use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKRegressorPredictor,
    sk_run_operation,
};

use super::super::error::SKBaggingError;
use super::super::fit::sk_map_row_chunks;
use super::super::model::SKBaggingRegressorModel;
use super::super::validation::{
    sk_bagging_predict_dimensions, sk_validate_bagging_finite_features,
};
use super::regressor_routing::sk_predict_regressor_chunk;

impl<F, M> SKBaggingRegressorModel<F, M>
where
    F: SKFloat,
    M: SKRegressorPredictor<F> + Send + Sync,
    M::Error: Error + Send + 'static,
{
    /// Predict the mean response across all fitted base estimators for every row.
    pub fn predict_values<'a, X>(&self, features: X) -> Result<Array1<F>, SKBaggingError<M::Error>>
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
                F::from(self.estimators.len()).expect("estimator count fits in scalar");
            let chunks = sk_map_row_chunks(
                rows_count,
                self.estimators.len(),
                self.execution_intent,
                |row_start, row_end| {
                    let mut sum = Array1::<F>::zeros(row_end - row_start);
                    self.estimators
                        .iter()
                        .zip(self.feature_subsets.iter())
                        .try_for_each(|(estimator, feature_subset)| {
                            let part = sk_predict_regressor_chunk(
                                estimator,
                                &data,
                                row_start,
                                row_end,
                                feature_subset,
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

            let mut output = Array1::<F>::zeros(rows_count);
            chunks.into_iter().for_each(|(row_start, row_end, part)| {
                output
                    .slice_axis_mut(Axis(0), Slice::from(row_start..row_end))
                    .assign(&part);
            });
            Ok(output)
        })
    }
}

impl<F, M> SKRegressorPredictor<F> for SKBaggingRegressorModel<F, M>
where
    F: SKFloat,
    M: SKRegressorPredictor<F> + Send + Sync,
    M::Error: Error + Send + 'static,
{
    type Error = SKBaggingError<M::Error>;

    fn predict<'a, X>(&self, features: X) -> Result<Array1<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        self.predict_values(features)
    }
}
