//! Decision-function surface for [`SKSVCModel`]
//! (spec `support-vector-machines`).

use ndarray::{Array2, Axis};
use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, sk_run_operation,
};

use super::model::SKSVCModel;
use super::predict::sk_pairwise_decision_matrix;
use crate::kernel::sk_svm_view_dimensions;

/// Aggregate `c(c - 1)/2` pairwise margins into a `(rows, c)` OvO confidence matrix.
fn sk_aggregate_ovo_confidences<F: SKFloat>(
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
                .zip(votes.into_iter().zip(sum_margins.into_iter()))
                .for_each(|(cell, (vote_count, margin_sum))| {
                    let scaled = margin_sum / (three * (margin_sum.abs() + F::one()));
                    *cell = F::from(vote_count).expect("vote count fits scalar") + scaled;
                });
        });
    output
}

impl<F: SKFloat> SKSVCModel<F> {
    /// Compute the decision function: `(m, 1)` signed margin for binary
    /// problems and `(m, c)` OvO-aggregated per-class confidences for
    /// multiclass problems.
    pub fn decision_function<'a, X>(&self, features: X) -> Result<Array2<F>, SKError>
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
            if self.number_of_classes() == 2 {
                return Ok(pairwise);
            }
            let pairs: Vec<(usize, usize)> = self
                .pairwise_models
                .iter()
                .map(|head| (head.class_negative, head.class_positive))
                .collect();
            Ok(sk_aggregate_ovo_confidences(
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
        let (rows_count, columns_count) =
            sk_svm_view_dimensions(&features).unwrap_or((0, 0));
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Predict,
            rows_count,
            columns_count,
            execution_mode: self.execution_mode,
            backend: self.backend,
        };
        sk_run_operation(attributes, || sk_pairwise_decision_matrix(self, features))
    }
}
