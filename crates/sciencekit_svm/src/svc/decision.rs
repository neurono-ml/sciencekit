//! Decision-function surface for [`SKSVCModel`]
//! (spec `support-vector-machines`).

use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;
use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, sk_run_operation,
};
use sciencekit_math::{
    SKKernel, sk_kernel_matrix, sk_kernel_matrix_sequential, sk_validate_precomputed_query_kernel,
};

use super::model::{SKBinarySVCModel, SKSVCModel};
use crate::kernel::{
    sk_resolve_svm_execution_plan, sk_svm_view_dimensions, sk_validate_svm_query_features,
};

/// Evaluate one binary C-SVC head across all query rows.
pub(crate) fn sk_evaluate_binary_head<F: SKFloat>(
    head: &SKBinarySVCModel<F>,
    model: &SKSVCModel<F>,
    features: SKDataView<'_, F>,
    rows_count: usize,
    allow_parallel: bool,
) -> Result<Array1<F>, SKError> {
    if head.dual_coefficients.is_empty() {
        return Ok(Array1::from_elem(rows_count, head.bias));
    }
    if model.kernel_parameters.kernel == SKKernel::Precomputed {
        let query_gram = sk_validate_precomputed_query_kernel(features, model.number_of_features)?;
        let values: Vec<F> = (0..rows_count)
            .map(|row| {
                head.support_training_indices
                    .iter()
                    .zip(head.dual_coefficients.iter())
                    .fold(head.bias, |accumulator, (&support_index, &coefficient)| {
                        accumulator + coefficient * query_gram[[row, support_index]]
                    })
            })
            .collect();
        return Ok(Array1::from(values));
    }
    let kernel_values = if allow_parallel {
        sk_kernel_matrix(
            &model.kernel_parameters,
            features,
            head.support_vectors.view(),
        )?
    } else {
        sk_kernel_matrix_sequential(
            &model.kernel_parameters,
            features,
            head.support_vectors.view(),
        )?
    };
    let values: Vec<F> = if allow_parallel && rows_count >= 256 {
        kernel_values
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|row| row.dot(&head.dual_coefficients) + head.bias)
            .collect()
    } else {
        kernel_values
            .axis_iter(Axis(0))
            .map(|row| row.dot(&head.dual_coefficients) + head.bias)
            .collect()
    };
    Ok(Array1::from(values))
}

/// Evaluate all `c(c - 1)/2` pairwise binary decision columns `(rows, pairs)`.
pub(crate) fn sk_pairwise_decision_matrix<F: SKFloat>(
    model: &SKSVCModel<F>,
    features: SKDataView<'_, F>,
) -> Result<Array2<F>, SKError> {
    let (rows_count, columns_count) = sk_validate_svm_query_features(
        features,
        model.kernel_parameters.kernel,
        model.number_of_features,
        "predict",
    )?;
    let plan =
        sk_resolve_svm_execution_plan(&model.builder_state, &features, rows_count, columns_count)?;
    let allow_parallel = plan.parallelism > 1;
    let mut matrix = Array2::zeros((rows_count, model.pairwise_models.len()));
    for (pair_index, head) in model.pairwise_models.iter().enumerate() {
        let column = sk_evaluate_binary_head(head, model, features, rows_count, allow_parallel)?;
        matrix.column_mut(pair_index).assign(&column);
    }
    Ok(matrix)
}

/// Aggregate `c(c - 1)/2` pairwise margins into a `(rows, c)` one-vs-one confidence matrix.
fn sk_aggregate_one_vs_one_confidences<F: SKFloat>(
    pairwise: &Array2<F>,
    pairs: &[(usize, usize)],
    number_of_classes: usize,
) -> Array2<F> {
    let rows_count = pairwise.nrows();
    let three = F::from(3.0).expect("3.0 fits SKFloat");
    let mut output = Array2::zeros((rows_count, number_of_classes));

    output
        .axis_iter_mut(Axis(0))
        .zip(pairwise.axis_iter(Axis(0)))
        .for_each(|(mut out_row, pair_row)| {
            let mut votes = vec![0usize; number_of_classes];
            let mut sum_margins = vec![F::zero(); number_of_classes];
            pair_row.iter().zip(pairs.iter()).for_each(
                |(&margin, &(class_negative, class_positive))| {
                    if margin > F::zero() {
                        votes[class_positive] += 1;
                    } else {
                        votes[class_negative] += 1;
                    }
                    sum_margins[class_positive] = sum_margins[class_positive] + margin;
                    sum_margins[class_negative] = sum_margins[class_negative] - margin;
                },
            );
            out_row
                .iter_mut()
                .zip(votes.into_iter().zip(sum_margins))
                .for_each(|(cell, (vote_count, margin_sum))| {
                    let scaled = margin_sum / (three * (margin_sum.abs() + F::one()));
                    *cell = F::from(vote_count).expect("vote count fits scalar") + scaled;
                });
        });
    output
}

impl<F: SKFloat> SKSVCModel<F> {
    /// Compute the decision function: `(m, 1)` signed margin for binary
    /// problems and `(m, c)` one-vs-one aggregated per-class confidences for
    /// multiclass problems.
    pub fn decision_function<'a, X>(&self, features: X) -> Result<Array2<F>, SKError>
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
            let pairwise = sk_pairwise_decision_matrix(self, features)?;
            if self.number_of_classes() == 2 {
                return Ok(pairwise);
            }
            let pairs: Vec<(usize, usize)> = self
                .pairwise_models
                .iter()
                .map(|head| (head.class_negative, head.class_positive))
                .collect();
            Ok(sk_aggregate_one_vs_one_confidences(
                &pairwise,
                &pairs,
                self.number_of_classes(),
            ))
        })
    }

    /// Compute the raw `(m, c(c-1)/2)` pairwise binary margins in `(a, b)` order.
    pub fn pairwise_decision_function<'a, X>(&self, features: X) -> Result<Array2<F>, SKError>
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
        sk_run_operation(attributes, || sk_pairwise_decision_matrix(self, features))
    }
}
