//! The linear-regression adapter fit: a monotone descent on the mean squared
//! error of the frozen base plus the low-rank update.

use ndarray::{Array2, ArrayView1, ArrayView2, Axis};
use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKTargetView,
    sk_run_operation,
};

use super::gradients::sk_transposed_feature_product;
use super::validation::{sk_validate_finite_features, sk_validate_finite_targets};
use crate::adapters::lora::attached_model::SKLoraAttachedModel;
use crate::adapters::lora::prediction::{sk_feature_dimensions, sk_single_head_scores};

/// How many times a rejected step is halved before it is accepted as-is.
const SK_LORA_MAXIMUM_BACKTRACKING_ATTEMPTS: usize = 24;

/// The mean squared error of a linear adapter on the given data.
pub(super) fn sk_linear_loss<F: SKFloat>(
    data: &SKDataView<'_, F>,
    coefficients: ArrayView2<'_, F>,
    intercept: F,
    targets: ArrayView1<'_, F>,
) -> Result<F, SKError> {
    let prediction = sk_single_head_scores(data, coefficients.row(0), intercept)?;
    let total = prediction
        .iter()
        .zip(targets.iter())
        .map(|(&prediction, &target)| (prediction - target) * (prediction - target))
        .fold(F::zero(), |accumulator, term| accumulator + term);
    Ok(total / F::from(targets.len()).unwrap_or_else(F::one))
}

impl<F: SKFloat> SKLoraAttachedModel<F> {
    /// Fit the factors on continuous targets, leaving the base untouched.
    pub(super) fn fit_linear(
        &mut self,
        data: &SKDataView<'_, F>,
        targets: &SKTargetView<'_, F>,
    ) -> Result<(), SKError> {
        let continuous = targets.as_continuous()?;
        let (rows_count, columns_count) = sk_feature_dimensions(data);
        let expected = self.feature_count();
        if rows_count == 0 || columns_count == 0 {
            return Err(SKError::ShapeMismatch {
                expected: vec![rows_count.max(1), expected.max(1)],
                found: vec![rows_count, columns_count],
            });
        }
        if columns_count != expected {
            return Err(SKError::shape_mismatch_2d(
                rows_count,
                expected,
                rows_count,
                columns_count,
            ));
        }
        if continuous.len() != rows_count {
            return Err(SKError::ShapeMismatch {
                expected: vec![rows_count],
                found: vec![continuous.len()],
            });
        }
        sk_validate_finite_features(data)?;
        sk_validate_finite_targets(continuous.view())?;

        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Fit,
            rows_count,
            columns_count,
            execution_mode: self.adapter().execution_mode(),
            backend: self.adapter().backend(),
        };
        sk_run_operation(attributes, || self.descend_linear(data, continuous.view()))
    }

    fn descend_linear(
        &mut self,
        data: &SKDataView<'_, F>,
        targets: ArrayView1<'_, F>,
    ) -> Result<(), SKError> {
        let base_coefficients = self.base().coefficients()?;
        let intercept = self.base().intercepts()?[0];
        let scaling = self.adapter().scaling();
        let mut loss = sk_linear_loss(
            data,
            (&base_coefficients + &self.delta()).view(),
            intercept,
            targets,
        )?;
        let mut step = self.adapter().learning_rate();
        for _ in 0..self.adapter().maximum_number_of_iterations() {
            let (gradient_a, gradient_b) =
                self.linear_gradient(data, targets, &base_coefficients)?;
            let (candidate_a, candidate_b, candidate_loss, accepted_step) = self
                .linear_backtracking_step(
                    data,
                    targets,
                    &base_coefficients,
                    intercept,
                    scaling,
                    &gradient_a,
                    &gradient_b,
                    loss,
                    step,
                )?;
            step = accepted_step;
            let relative_decrease = (loss - candidate_loss) / loss.max(F::epsilon());
            self.a_matrix = candidate_a;
            self.b_matrix = candidate_b;
            loss = candidate_loss;
            if relative_decrease < self.adapter().tolerance() {
                break;
            }
        }
        Ok(())
    }

    /// The gradients of the mean squared error with respect to `A` and `B`.
    fn linear_gradient(
        &self,
        data: &SKDataView<'_, F>,
        targets: ArrayView1<'_, F>,
        base_coefficients: &Array2<F>,
    ) -> Result<(Array2<F>, Array2<F>), SKError> {
        let intercept = self.base().intercepts()?[0];
        let scaling = self.adapter().scaling();
        let effective = base_coefficients + &self.delta();
        let prediction = sk_single_head_scores(data, effective.row(0), intercept)?;
        let rows = F::from(targets.len()).unwrap_or_else(F::one);
        let factor = F::from(2.0).unwrap_or_else(F::one) / rows;
        let score_gradient = (prediction - targets).mapv(|value| value * factor);
        let delta_gradient =
            sk_transposed_feature_product(data, score_gradient.insert_axis(Axis(1)).view())?;
        let column = delta_gradient.column(0);
        let gradient_b = self
            .a_matrix
            .dot(&column)
            .insert_axis(Axis(0))
            .mapv(|value| value * scaling);
        let gradient_a = self
            .b_matrix
            .t()
            .dot(&column.insert_axis(Axis(0)))
            .mapv(|value| value * scaling);
        Ok((gradient_a, gradient_b))
    }

    /// Try decreasing steps until the loss strictly decreases.
    #[allow(clippy::too_many_arguments)]
    fn linear_backtracking_step(
        &self,
        data: &SKDataView<'_, F>,
        targets: ArrayView1<'_, F>,
        base_coefficients: &Array2<F>,
        intercept: F,
        scaling: F,
        gradient_a: &Array2<F>,
        gradient_b: &Array2<F>,
        loss: F,
        initial_step: F,
    ) -> Result<(Array2<F>, Array2<F>, F, F), SKError> {
        let mut step = initial_step;
        for attempt in 0..=SK_LORA_MAXIMUM_BACKTRACKING_ATTEMPTS {
            let candidate_a = &self.a_matrix - &gradient_a.mapv(|value| value * step);
            let candidate_b = &self.b_matrix - &gradient_b.mapv(|value| value * step);
            let candidate_coefficients =
                base_coefficients + &candidate_b.dot(&candidate_a).mapv(|value| value * scaling);
            let candidate_loss =
                sk_linear_loss(data, candidate_coefficients.view(), intercept, targets)?;
            if candidate_loss < loss || attempt == SK_LORA_MAXIMUM_BACKTRACKING_ATTEMPTS {
                return Ok((candidate_a, candidate_b, candidate_loss, step));
            }
            step = step * F::from(0.5).unwrap_or_else(F::one);
        }
        Ok((self.a_matrix.clone(), self.b_matrix.clone(), loss, step))
    }
}
