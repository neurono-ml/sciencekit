//! The logistic-regression adapter fit: descent on the base's logistic
//! objective while the base parameters stay frozen.

use ndarray::{Array1, Array2, ArrayView1};
use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKTargetView,
    sk_run_operation,
};

use super::gradients::sk_transposed_feature_product;
use super::validation::sk_validate_finite_features;
use crate::adapters::lora::attached_model::SKLoraAttachedModel;
use crate::adapters::lora::base_model::SKLoraBaseModel;
use crate::adapters::lora::prediction::{sk_feature_dimensions, sk_multi_head_scores};
use crate::logistic::objective::{sk_log_sigmoid, sk_multinomial_loss};

/// How many times a rejected step is halved before it is accepted as-is.
const SK_LORA_MAXIMUM_BACKTRACKING_ATTEMPTS: usize = 24;

/// The labels as canonical class indices.
pub(super) fn sk_integer_labels<'a, F: SKFloat>(
    targets: &SKTargetView<'a, F>,
) -> Result<ArrayView1<'a, i64>, SKError> {
    match targets {
        SKTargetView::Integer(labels) => Ok(*labels),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "continuous targets",
            suggestion: "pass canonical integer class indices to a classifier adapter",
        }),
    }
}

/// The mean logistic loss of the adapter at its current factors.
pub(super) fn sk_logistic_loss<F: SKFloat>(
    attached: &SKLoraAttachedModel<F>,
    data: &SKDataView<'_, F>,
    labels: ArrayView1<'_, i64>,
) -> Result<F, SKError> {
    let (rows_count, _) = sk_feature_dimensions(data);
    let scores = attached.decision_function_from_view(data)?;
    let heads = scores.ncols();
    let total = (0..rows_count).try_fold(F::zero(), |accumulator, row| {
        let label = labels[row];
        if label < 0 || label as usize >= heads.max(2) {
            return Err(SKError::UnknownLabel { label });
        }
        let row_score = scores.row(row);
        let term = if heads == 1 {
            let logit = row_score[0];
            match label {
                1 => -sk_log_sigmoid(logit),
                0 => -sk_log_sigmoid(-logit),
                other => return Err(SKError::UnknownLabel { label: other }),
            }
        } else {
            let logits = row_score
                .as_slice()
                .ok_or(SKError::UnsupportedRepresentation {
                    representation: "non-contiguous decision row",
                    suggestion: "reshape the decision scores to a contiguous matrix",
                })?;
            sk_multinomial_loss(logits, label as usize, F::one())
        };
        Ok(accumulator + term)
    })?;
    Ok(total / F::from(rows_count).unwrap_or_else(F::one))
}

impl<F: SKFloat> SKLoraAttachedModel<F> {
    /// Fit the factors on integer class labels, leaving the base untouched.
    pub(super) fn fit_logistic(
        &mut self,
        data: &SKDataView<'_, F>,
        targets: &SKTargetView<'_, F>,
    ) -> Result<(), SKError> {
        let labels = sk_integer_labels(targets)?;
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
        if labels.len() != rows_count {
            return Err(SKError::ShapeMismatch {
                expected: vec![rows_count],
                found: vec![labels.len()],
            });
        }
        sk_validate_finite_features(data)?;

        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Fit,
            rows_count,
            columns_count,
            execution_mode: self.adapter().execution_mode(),
            backend: self.adapter().backend(),
        };
        sk_run_operation(attributes, || self.descend_logistic(data, labels))
    }

    fn descend_logistic(
        &mut self,
        data: &SKDataView<'_, F>,
        labels: ArrayView1<'_, i64>,
    ) -> Result<(), SKError> {
        let base_coefficients = self.base().coefficients()?;
        let intercepts = self.base().intercepts()?;
        let scaling = self.adapter().scaling();
        let mut loss = sk_logistic_loss(self, data, labels)?;
        let mut step = self.adapter().learning_rate();
        for _ in 0..self.adapter().maximum_number_of_iterations() {
            let (gradient_a, gradient_b) =
                self.logistic_gradient(data, labels, &base_coefficients, &intercepts)?;
            let (candidate_a, candidate_b, candidate_loss, accepted_step) = self
                .logistic_backtracking_step(
                    data,
                    labels,
                    &base_coefficients,
                    &intercepts,
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

    /// The gradient of the mean logistic loss with respect to `A` and `B`.
    fn logistic_gradient(
        &self,
        data: &SKDataView<'_, F>,
        labels: ArrayView1<'_, i64>,
        base_coefficients: &Array2<F>,
        intercepts: &Array1<F>,
    ) -> Result<(Array2<F>, Array2<F>), SKError> {
        let scaling = self.adapter().scaling();
        let effective = base_coefficients + &self.delta();
        let scores = sk_multi_head_scores(data, effective.view(), intercepts.view())?;
        let probabilities = match self.base() {
            SKLoraBaseModel::LogisticRegression(model) => model.probabilities_from_scores(scores),
            _ => {
                return Err(SKError::UnsupportedOperation {
                    operation: "fit",
                    reason: "only a logistic-regression base has the logistic loss".to_string(),
                });
            }
        };
        let rows = F::from(labels.len()).unwrap_or_else(F::one);
        let heads = self.number_of_heads();
        let mut score_gradient = Array2::<F>::zeros((labels.len(), heads));
        for (row, &label) in labels.iter().enumerate() {
            if heads == 1 {
                let target = if label == 1 {
                    F::one()
                } else if label == 0 {
                    F::zero()
                } else {
                    return Err(SKError::UnknownLabel { label });
                };
                let position = heads.saturating_sub(1);
                score_gradient[[row, position]] = (probabilities[[row, 1]] - target) / rows;
            } else {
                if label < 0 || label as usize >= heads {
                    return Err(SKError::UnknownLabel { label });
                }
                for head in 0..heads {
                    let target = if head == label as usize {
                        F::one()
                    } else {
                        F::zero()
                    };
                    score_gradient[[row, head]] = (probabilities[[row, head]] - target) / rows;
                }
            }
        }
        let delta_gradient = sk_transposed_feature_product(data, score_gradient.view())?
            .t()
            .to_owned();
        let gradient_b = delta_gradient
            .dot(&self.a_matrix.t())
            .mapv(|value| value * scaling);
        let gradient_a = self
            .b_matrix
            .t()
            .dot(&delta_gradient)
            .mapv(|value| value * scaling);
        Ok((gradient_a, gradient_b))
    }

    /// Try decreasing steps until the loss strictly decreases.
    #[allow(clippy::too_many_arguments)]
    fn logistic_backtracking_step(
        &self,
        data: &SKDataView<'_, F>,
        labels: ArrayView1<'_, i64>,
        base_coefficients: &Array2<F>,
        intercepts: &Array1<F>,
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
            let candidate_delta = candidate_b.dot(&candidate_a).mapv(|value| value * scaling);
            let candidate_coefficients = base_coefficients + &candidate_delta;
            let scores =
                sk_multi_head_scores(data, candidate_coefficients.view(), intercepts.view())?;
            let candidate_loss = match self.base() {
                SKLoraBaseModel::LogisticRegression(model) => {
                    let probabilities = model.probabilities_from_scores(scores);
                    sk_loss_from_probabilities(&probabilities, labels)?
                }
                _ => {
                    return Err(SKError::UnsupportedOperation {
                        operation: "fit",
                        reason: "only a logistic-regression base has the logistic loss".to_string(),
                    });
                }
            };
            if candidate_loss < loss || attempt == SK_LORA_MAXIMUM_BACKTRACKING_ATTEMPTS {
                return Ok((candidate_a, candidate_b, candidate_loss, step));
            }
            step = step * F::from(0.5).unwrap_or_else(F::one);
        }
        Ok((self.a_matrix.clone(), self.b_matrix.clone(), loss, step))
    }

    /// Decision scores from an already converted feature view.
    pub(super) fn decision_function_from_view(
        &self,
        data: &SKDataView<'_, F>,
    ) -> Result<Array2<F>, SKError> {
        let coefficients = self.effective_coefficients()?;
        let intercepts = self.base().intercepts()?;
        sk_multi_head_scores(data, coefficients.view(), intercepts.view())
    }
}

/// The mean cross-entropy from probabilities, safely clamped at zero.
fn sk_loss_from_probabilities<F: SKFloat>(
    probabilities: &Array2<F>,
    labels: ArrayView1<'_, i64>,
) -> Result<F, SKError> {
    let rows = F::from(labels.len()).unwrap_or_else(F::one);
    let minimum = F::from(1e-15).unwrap_or_else(F::epsilon);
    let total = labels
        .iter()
        .enumerate()
        .try_fold(F::zero(), |sum, (row, &label)| {
            let heads = probabilities.ncols();
            let index = label as usize;
            if index >= heads {
                return Err(SKError::UnknownLabel { label });
            }
            let probability = probabilities[[row, index]].max(minimum);
            Ok(sum - probability.ln())
        })?;
    Ok(total / rows)
}
