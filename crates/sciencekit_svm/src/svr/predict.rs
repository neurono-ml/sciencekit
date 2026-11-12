//! `SKRegressorPredictor` implementation for [`SKSVRModel`]
//! (spec `support-vector-machines`).

use ndarray::{Array1, Axis};
use rayon::prelude::*;
use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKRegressorPredictor,
    sk_run_operation,
};
use sciencekit_math::{
    SKKernel, sk_kernel_matrix, sk_kernel_matrix_sequential, sk_validate_precomputed_query_kernel,
};

use super::model::SKSVRModel;
use crate::kernel::{
    sk_resolve_svm_execution_plan, sk_svm_view_dimensions, sk_validate_svm_query_features,
};

impl<F: SKFloat> SKRegressorPredictor<F> for SKSVRModel<F> {
    type Error = SKError;

    fn predict<'a, X>(&self, features: X) -> Result<Array1<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        let (rows_count, columns_count) = sk_svm_view_dimensions(&features).unwrap_or((0, 0));
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Predict,
            rows_count,
            columns_count,
            execution_mode: self.execution_mode(),
            backend: self.backend,
        };
        sk_run_operation(attributes, || {
            let (validated_rows, validated_columns) = sk_validate_svm_query_features(
                features,
                self.kernel_parameters.kernel,
                self.number_of_features,
                "predict",
            )?;
            let plan = sk_resolve_svm_execution_plan(
                &self.builder_state,
                &features,
                validated_rows,
                validated_columns,
            )?;
            let allow_parallel = plan.parallelism > 1;
            if self.dual_coefficients.is_empty() {
                return Ok(Array1::from_elem(validated_rows, self.bias));
            }
            if self.kernel_parameters.kernel == SKKernel::Precomputed {
                let query_gram =
                    sk_validate_precomputed_query_kernel(features, self.number_of_features)?;
                let values: Vec<F> = (0..validated_rows)
                    .map(|row| {
                        self.support_training_indices
                            .iter()
                            .zip(self.dual_coefficients.iter())
                            .fold(self.bias, |accumulator, (&support_index, &coefficient)| {
                                accumulator + coefficient * query_gram[[row, support_index]]
                            })
                    })
                    .collect();
                return Ok(Array1::from(values));
            }

            let kernel_values = if allow_parallel {
                sk_kernel_matrix(
                    &self.kernel_parameters,
                    features,
                    self.support_vectors.view(),
                )?
            } else {
                sk_kernel_matrix_sequential(
                    &self.kernel_parameters,
                    features,
                    self.support_vectors.view(),
                )?
            };
            let values: Vec<F> = if allow_parallel && validated_rows >= 256 {
                kernel_values
                    .axis_iter(Axis(0))
                    .into_par_iter()
                    .map(|row| row.dot(&self.dual_coefficients) + self.bias)
                    .collect()
            } else {
                kernel_values
                    .axis_iter(Axis(0))
                    .map(|row| row.dot(&self.dual_coefficients) + self.bias)
                    .collect()
            };
            Ok(Array1::from(values))
        })
    }
}
