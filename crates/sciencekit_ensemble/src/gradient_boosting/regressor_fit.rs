//! Fitting entry point for [`SKGradientBoostingRegressor`].

use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKSupervisedFit,
    SKTargetView, sk_run_operation,
};

use crate::boosting::loss::{SKAbsoluteErrorLoss, SKHuberLoss, SKSquaredErrorLoss};
use crate::boosting::scalar_loop::sk_run_scalar_boosting;

use super::importances::sk_aggregate_stage_importances;
use super::loss_kind::SKGradientBoostingRegressionLoss;
use super::regressor::SKGradientBoostingRegressor;
use super::regressor_model::SKGradientBoostingRegressorModel;
use super::validation::{sk_fit_dimensions, sk_reject_streaming_mode, sk_validate_finite_features};

impl<F: SKFloat> SKSupervisedFit<F> for SKGradientBoostingRegressor<F> {
    type Model = SKGradientBoostingRegressorModel<F>;
    type Error = SKError;

    fn fit<'a, X, T>(&self, features: X, targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        sk_fit_gradient_boosting_regressor(self, features.try_into()?, targets.try_into()?)
    }
}

pub(super) fn sk_fit_gradient_boosting_regressor<F: SKFloat>(
    estimator: &SKGradientBoostingRegressor<F>,
    features: SKDataView<'_, F>,
    targets: SKTargetView<'_, F>,
) -> Result<SKGradientBoostingRegressorModel<F>, SKError> {
    let (rows_count, columns_count) = sk_fit_dimensions(&features)?;
    let attributes = SKOperationAttributes {
        operation: SKOperationKind::Fit,
        rows_count,
        columns_count,
        execution_mode: estimator.execution_intent(),
        backend: estimator.backend(),
    };
    sk_run_operation(attributes, || {
        sk_reject_streaming_mode(estimator.execution_intent())?;
        sk_validate_finite_features(&features)?;
        let responses = targets.as_continuous()?;
        if responses.len() != rows_count {
            return Err(SKError::shape_mismatch_2d(
                rows_count,
                1,
                responses.len(),
                1,
            ));
        }
        if let Some(position) = responses.iter().position(|value| !value.is_finite()) {
            return Err(SKError::NonFiniteValue { position });
        }
        let target_values: Vec<F> = responses.iter().copied().collect();
        let parameters = estimator.configuration.stage_parameters();
        let early_stopping = estimator.configuration.early_stopping_config();
        let fitted = match estimator.loss() {
            SKGradientBoostingRegressionLoss::SquaredError => sk_run_scalar_boosting(
                &SKSquaredErrorLoss::new(),
                &features,
                &target_values,
                estimator.number_of_estimators(),
                &parameters,
                early_stopping,
            )?,
            SKGradientBoostingRegressionLoss::AbsoluteError => sk_run_scalar_boosting(
                &SKAbsoluteErrorLoss::new(),
                &features,
                &target_values,
                estimator.number_of_estimators(),
                &parameters,
                early_stopping,
            )?,
            SKGradientBoostingRegressionLoss::Huber => {
                let huber_loss = SKHuberLoss::build()
                    .delta(estimator.huber_delta())
                    .build()?;
                sk_run_scalar_boosting(
                    &huber_loss,
                    &features,
                    &target_values,
                    estimator.number_of_estimators(),
                    &parameters,
                    early_stopping,
                )?
            }
        };
        let feature_importances =
            sk_aggregate_stage_importances(fitted.stages.iter(), columns_count);
        let stages = fitted.stages.into_iter().map(|slot| slot.tree).collect();
        Ok(SKGradientBoostingRegressorModel {
            stages,
            initial_prediction: fitted.initial_prediction,
            learning_rate: estimator.learning_rate(),
            feature_count: columns_count,
            feature_importances,
            loss: estimator.loss(),
            huber_delta: estimator.huber_delta(),
            backend: estimator.backend(),
            execution_intent: estimator.execution_intent(),
        })
    })
}
