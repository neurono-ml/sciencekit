//! Fitting entry point for [`SKHistGradientBoostingRegressor`].

use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKSupervisedFit,
    SKTargetView, sk_run_operation,
};

use crate::boosting::loss::{SKAbsoluteErrorLoss, SKSquaredErrorLoss};
use crate::histogram_boosting::binning::SKBinnedDataset;
use crate::histogram_boosting::fit::{sk_aggregate_histogram_importances, sk_validation_rows};
use crate::histogram_boosting::scalar_loop::sk_run_histogram_scalar;

use super::loss_kind::SKHistGradientBoostingRegressionLoss;
use super::regressor::SKHistGradientBoostingRegressor;
use super::regressor_model::SKHistGradientBoostingRegressorModel;
use super::validation::{
    sk_fit_dimensions, sk_reject_streaming_mode, sk_validate_finite_or_missing_features,
};

impl<F: SKFloat> SKSupervisedFit<F> for SKHistGradientBoostingRegressor<F> {
    type Model = SKHistGradientBoostingRegressorModel<F>;
    type Error = SKError;

    fn fit<'a, X, T>(&self, features: X, targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        sk_fit_histogram_regressor(self, features.try_into()?, targets.try_into()?)
    }
}

pub(super) fn sk_fit_histogram_regressor<F: SKFloat>(
    estimator: &SKHistGradientBoostingRegressor<F>,
    features: SKDataView<'_, F>,
    targets: SKTargetView<'_, F>,
) -> Result<SKHistGradientBoostingRegressorModel<F>, SKError> {
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
        sk_validate_finite_or_missing_features(&features)?;
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
        let dataset = SKBinnedDataset::build(
            &features,
            estimator.categorical_features(),
            estimator.maximum_bins(),
        )?;
        let parameters = estimator.configuration.stage_parameters();
        let early_stopping = estimator.configuration.early_stopping_config();
        let (training_rows, validation_rows) =
            sk_validation_rows(rows_count, None, 1, early_stopping);
        let fitted = match estimator.loss() {
            SKHistGradientBoostingRegressionLoss::SquaredError => sk_run_histogram_scalar(
                &SKSquaredErrorLoss::new(),
                &dataset,
                &target_values,
                &training_rows,
                validation_rows.as_deref(),
                estimator.number_of_estimators(),
                &parameters,
                early_stopping,
            )?,
            SKHistGradientBoostingRegressionLoss::AbsoluteError => sk_run_histogram_scalar(
                &SKAbsoluteErrorLoss::new(),
                &dataset,
                &target_values,
                &training_rows,
                validation_rows.as_deref(),
                estimator.number_of_estimators(),
                &parameters,
                early_stopping,
            )?,
        };
        let tree_refs: Vec<_> = fitted.stages.iter().collect();
        let feature_importances = sk_aggregate_histogram_importances(
            &fitted.stage_importances,
            &tree_refs,
            columns_count,
        );
        Ok(SKHistGradientBoostingRegressorModel {
            stages: fitted.stages,
            stage_importances: fitted.stage_importances,
            initial_prediction: fitted.initial_prediction,
            learning_rate: estimator.learning_rate(),
            feature_count: columns_count,
            feature_importances,
            feature_binnings: dataset.feature_binnings().to_vec(),
            loss: estimator.loss(),
            backend: estimator.backend(),
            execution_intent: estimator.execution_intent(),
        })
    })
}
