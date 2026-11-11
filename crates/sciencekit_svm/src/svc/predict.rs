//! Prediction and decision surfaces on [`SKSVCModel`]
//! (spec `support-vector-machines`).

use ndarray::{Array1, Array2, Axis};
use rayon::prelude::*;
use sciencekit_common::{
    SKClassifierPredictor, SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind,
    sk_run_operation,
};
use sciencekit_math::{SKKernel, sk_kernel_matrix, sk_validate_precomputed_query_kernel};

use super::model::{SKBinarySVCModel, SKSVCModel};
use crate::kernel::{sk_svm_view_dimensions, sk_validate_svm_query_features};

/// Resolve canonical winner class from pairwise decisions in `(a, b)` order
/// (`a < b`, positive favors `b`), breaking vote ties by smallest class index.
pub fn sk_resolve_ovo_votes<F: SKFloat>(
    pairwise_decisions: &[F],
    pairs: &[(usize, usize)],
    number_of_classes: usize,
) -> usize {
    let mut votes = vec![0usize; number_of_classes];
    pairwise_decisions
        .iter()
        .zip(pairs.iter())
        .for_each(|(&decision, &(class_negative, class_positive))| {
            if decision > F::zero() {
                votes[class_positive] += 1;
            } else {
                votes[class_negative] += 1;
            }
        });
    let mut winner = 0usize;
    for class_index in 1..number_of_classes {
        if votes[class_index] > votes[winner] {
            winner = class_index;
        }
    }
    winner
}

/// Evaluate one binary C-SVC head across all query rows.
pub(crate) fn sk_evaluate_binary_head<F: SKFloat>(
    head: &SKBinarySVCModel<F>,
    model: &SKSVCModel<F>,
    features: SKDataView<'_, F>,
    rows_count: usize,
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
                    .fold(head.bias, |acc, (&sv_idx, &coef)| {
                        acc + coef * query_gram[[row, sv_idx]]
                    })
            })
            .collect();
        return Ok(Array1::from(values));
    }
    let kernel_values = sk_kernel_matrix(
        &model.kernel_parameters,
        features,
        head.support_vectors.view(),
    )?;
    let values: Vec<F> = if rows_count >= 256 {
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
    let (rows_count, _) =
        sk_validate_svm_query_features(features, model.number_of_features, "predict")?;
    let mut matrix = Array2::zeros((rows_count, model.pairwise_models.len()));
    for (pair_index, head) in model.pairwise_models.iter().enumerate() {
        let column = sk_evaluate_binary_head(head, model, features, rows_count)?;
        matrix.column_mut(pair_index).assign(&column);
    }
    Ok(matrix)
}

impl<F: SKFloat> SKClassifierPredictor<F> for SKSVCModel<F> {
    type Error = SKError;

    fn predict_labels<'a, X>(&self, features: X) -> Result<Array1<i64>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        let (rows_count, columns_count) =
            sk_svm_view_dimensions(&features).unwrap_or((0, 0));
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Predict,
            rows_count,
            columns_count,
            execution_mode: self.execution_mode,
            backend: self.backend,
        };
        sk_run_operation(attributes, || {
            let pairwise = sk_pairwise_decision_matrix(self, features)?;
            let pairs: Vec<(usize, usize)> = self
                .pairwise_models
                .iter()
                .map(|head| (head.class_negative, head.class_positive))
                .collect();
            let number_of_classes = self.number_of_classes();
            let labels: Vec<i64> = pairwise
                .axis_iter(Axis(0))
                .map(|row| {
                    let slice = row.to_vec();
                    sk_resolve_ovo_votes(&slice, &pairs, number_of_classes) as i64
                })
                .collect();
            Ok(Array1::from(labels))
        })
    }

    fn predict_probabilities<'a, X>(&self, _features: X) -> Result<Array2<F>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        Err(SKError::UnsupportedOperation {
            operation: "predict_probabilities",
            reason: "enable probability(true) on SKSVCBuilder before fitting".to_owned(),
        })
    }
}
