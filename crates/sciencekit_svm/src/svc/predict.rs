//! Prediction surface (`predict_labels`, `predict_probabilities`) on [`SKSVCModel`]
//! (spec `support-vector-machines`).

use ndarray::{Array1, Array2, Axis};
use sciencekit_common::{
    SKClassifierPredictor, SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind,
    sk_run_operation,
};

use super::decision::sk_pairwise_decision_matrix;
use super::model::SKSVCModel;
use crate::kernel::sk_svm_view_dimensions;
use crate::probability::sk_couple_pairwise_probabilities;

/// Resolve canonical winner class from pairwise decisions in `(a, b)` order
/// (`a < b`, positive favors `b`), breaking vote ties by smallest class index.
pub fn sk_resolve_one_vs_one_votes<F: SKFloat>(
    pairwise_decisions: &[F],
    pairs: &[(usize, usize)],
    number_of_classes: usize,
) -> usize {
    let mut votes = vec![0usize; number_of_classes];
    pairwise_decisions.iter().zip(pairs.iter()).for_each(
        |(&decision, &(class_negative, class_positive))| {
            if decision > F::zero() {
                votes[class_positive] += 1;
            } else {
                votes[class_negative] += 1;
            }
        },
    );
    votes
        .iter()
        .enumerate()
        .skip(1)
        .fold(0usize, |winner, (class_index, &count)| {
            if count > votes[winner] {
                class_index
            } else {
                winner
            }
        })
}

impl<F: SKFloat> SKClassifierPredictor<F> for SKSVCModel<F> {
    type Error = SKError;

    fn predict_labels<'a, X>(&self, features: X) -> Result<Array1<i64>, Self::Error>
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
                    sk_resolve_one_vs_one_votes(&slice, &pairs, number_of_classes) as i64
                })
                .collect();
            Ok(Array1::from(labels))
        })
    }

    fn predict_probabilities<'a, X>(&self, features: X) -> Result<Array2<F>, Self::Error>
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
            if !self.probability_enabled {
                return Err(SKError::UnsupportedOperation {
                    operation: "predict_probabilities",
                    reason: "enable probability(true) on SKSVCBuilder before fitting".to_owned(),
                });
            }
            let pairwise = sk_pairwise_decision_matrix(self, features)?;
            let number_of_classes = self.number_of_classes();
            let mut probabilities = Array2::zeros((pairwise.nrows(), number_of_classes));
            probabilities
                .axis_iter_mut(Axis(0))
                .zip(pairwise.axis_iter(Axis(0)))
                .for_each(|(mut out_row, margin_row)| {
                    let binary_positive: Vec<F> = margin_row
                        .iter()
                        .zip(self.pairwise_models.iter())
                        .map(|(&margin, head)| {
                            head.platt_parameters
                                .expect("calibrated when probability_enabled")
                                .predict_positive_probability(margin)
                        })
                        .collect();
                    let coupled =
                        sk_couple_pairwise_probabilities(&binary_positive, number_of_classes);
                    out_row
                        .iter_mut()
                        .zip(coupled)
                        .for_each(|(cell, probability)| *cell = probability);
                });
            Ok(probabilities)
        })
    }
}
