//! Fitting the low-rank factors of an attached adapter (spec `lora-adapters`).
//!
//! The base parameters never move: the optimizer touches only `A` and `B`. The
//! search is a deterministic, monotone gradient descent with a backtracking
//! step, so every accepted iteration strictly decreases the training loss.

use ndarray::{Array2, ArrayView1, ArrayView2, Axis, azip};
use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKTargetView,
    sk_run_operation,
};

use super::attached_model::SKLoraAttachedModel;
use super::base_model::SKLoraBaseModel;
use super::prediction::{sk_feature_dimensions, sk_single_head_scores};
use crate::logistic_regression::validation::unsupported_representation;

/// How many times a rejected step is halved before it is accepted as-is.
const SK_LORA_MAXIMUM_BACKTRACKING_ATTEMPTS: usize = 24;

/// The maximum mean squared error of a linear adapter on the given data.
fn sk_linear_loss<F: SKFloat>(
    data: &SKDataView<'_, F>,
    coefficients: ArrayView2<'_, F>,
    intercept: F,
    targets: ArrayView1<'_, F>,
) -> Result<F, SKError> {
    let prediction = sk_single_head_scores(data, coefficients.row(0), intercept)?;
    let mean_squared_error = prediction
        .iter()
        .zip(targets.iter())
        .map(|(&prediction, &target)| (prediction - target) * (prediction - target))
        .fold(F::zero(), |accumulator, term| accumulator + term)
        / F::from(targets.len()).unwrap_or_else(F::one);
    Ok(mean_squared_error)
}

/// `Xᵀ · gradient`, shaped `features × heads`, without densifying a sparse `X`.
fn sk_transposed_feature_product<F: SKFloat>(
    data: &SKDataView<'_, F>,
    gradient: ArrayView2<'_, F>,
) -> Result<Array2<F>, SKError> {
    match data {
        SKDataView::Dense(dense) => Ok(dense.t().dot(&gradient)),
        SKDataView::Sparse(sparse) => {
            let mut product = Array2::<F>::zeros((sparse.cols(), gradient.ncols()));
            for (row_index, sparse_row) in sparse.outer_iterator().enumerate() {
                let gradient_row = gradient.row(row_index);
                for (&column, &value) in sparse_row.indices().iter().zip(sparse_row.data()) {
                    let mut output_row = product.row_mut(column);
                    azip!((accumulator in &mut output_row, &term in &gradient_row) {
                        *accumulator = *accumulator + value * term;
                    });
                }
            }
            Ok(product)
        }
        _ => Err(unsupported_representation()),
    }
}

/// Reject an empty design, a feature-count disagreement and a target-length
/// disagreement before any arithmetic runs.
fn sk_validate_fit_shape(
    data: &SKDataView<'_, impl SKFloat>,
    feature_count: usize,
    target_count: usize,
) -> Result<usize, SKError> {
    let (rows_count, columns_count) = sk_feature_dimensions(data);
    if rows_count == 0 || columns_count == 0 {
        return Err(SKError::ShapeMismatch {
            expected: vec![rows_count.max(1), feature_count.max(1)],
            found: vec![rows_count, columns_count],
        });
    }
    if columns_count != feature_count {
        return Err(SKError::shape_mismatch_2d(
            rows_count,
            feature_count,
            rows_count,
            columns_count,
        ));
    }
    if target_count != rows_count {
        return Err(SKError::ShapeMismatch {
            expected: vec![rows_count],
            found: vec![target_count],
        });
    }
    Ok(rows_count)
}

/// Reject any non-finite feature or target, naming its flat position.
fn sk_validate_finite<F: SKFloat>(
    data: &SKDataView<'_, F>,
    targets: ArrayView1<'_, F>,
) -> Result<(), SKError> {
    let features_are_finite = match data {
        SKDataView::Dense(dense) => dense.iter().all(|value| value.is_finite()),
        SKDataView::Sparse(sparse) => sparse.data().iter().all(|value| value.is_finite()),
        _ => return Err(unsupported_representation()),
    };
    if !features_are_finite {
        return Err(SKError::NonFiniteValue { position: 0 });
    }
    if let Some(position) = targets.iter().position(|value| !value.is_finite()) {
        return Err(SKError::NonFiniteValue { position });
    }
    Ok(())
}

impl<F: SKFloat> SKLoraAttachedModel<F> {
    /// The mean squared error of a linear adapter on the given data.
    ///
    /// # Errors
    ///
    /// Returns a shape error when the features or targets disagree with the
    /// base, and [`SKError::UnsupportedOperation`] for a classifier base.
    pub fn training_loss<'a, X, T>(&self, features: X, targets: T) -> Result<F, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        let data = features.try_into()?;
        let targets = targets.try_into()?;
        let continuous = targets.as_continuous()?;
        sk_validate_fit_shape(&data, self.feature_count(), continuous.len())?;
        if !matches!(self.base(), SKLoraBaseModel::LinearRegression(_)) {
            return Err(SKError::UnsupportedOperation {
                operation: "training_loss",
                reason: "this adapter targets a classifier; its loss is the logistic loss"
                    .to_string(),
            });
        }
        sk_validate_finite(&data, continuous.view())?;
        let coefficients = self.effective_coefficients()?;
        let intercept = self.base().intercepts()?[0];
        sk_linear_loss(&data, coefficients.view(), intercept, continuous.view())
    }

    /// Fit the adapter on the supplied regression data.
    ///
    /// Only `A`/`B` move; the frozen base and its intercept stay untouched.
    ///
    /// # Errors
    ///
    /// Returns shape, non-finite and unsupported-family errors.
    pub fn fit<'a, X, T>(&mut self, features: X, targets: T) -> Result<&mut Self, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        let data = features.try_into()?;
        let targets = targets.try_into()?;
        match self.adapter().target() {
            super::config::SKLoraTargetModule::LinearCoefficients => {
                self.fit_linear(&data, &targets)?;
            }
            super::config::SKLoraTargetModule::LogisticCoefficients => {
                return Err(SKError::UnsupportedOperation {
                    operation: "fit",
                    reason: "logistic adapter fitting is not available yet".to_string(),
                });
            }
        }
        self.mark_fitted();
        Ok(self)
    }

    fn fit_linear(
        &mut self,
        data: &SKDataView<'_, F>,
        targets: &SKTargetView<'_, F>,
    ) -> Result<(), SKError> {
        let continuous = targets.as_continuous()?;
        sk_validate_fit_shape(data, self.feature_count(), continuous.len())?;
        sk_validate_finite(data, continuous.view())?;

        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Fit,
            rows_count: continuous.len(),
            columns_count: self.feature_count(),
            execution_mode: self.adapter().execution_mode(),
            backend: self.adapter().backend(),
        };
        sk_run_operation(attributes, || {
            let base_coefficients = self.base().coefficients()?;
            let intercept = self.base().intercepts()?[0];
            let scaling = self.adapter().scaling();
            let mut loss = sk_linear_loss(
                data,
                (&base_coefficients + &self.delta()).view(),
                intercept,
                continuous.view(),
            )?;
            let mut step = self.adapter().learning_rate();
            for _ in 0..self.adapter().maximum_number_of_iterations() {
                let (gradient_a, gradient_b) =
                    self.linear_gradient(data, continuous.view(), &base_coefficients)?;
                let (candidate_a, candidate_b, candidate_loss, accepted_step) = self
                    .backtracking_step(
                        data,
                        continuous.view(),
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
        })
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
        let score_gradient =
            (prediction - targets).mapv(|value| value * F::from(2.0).unwrap_or_else(F::one) / rows);
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
    fn backtracking_step(
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
