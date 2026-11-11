//! Fitting entry point for [`SKGradientBoostingClassifier`].

use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKSupervisedFit,
    SKTargetView, sk_run_operation,
};

use crate::boosting::loss::{SKBinaryLogLoss, SKMulticlassLogLoss};
use crate::boosting::multiclass_loop::sk_run_multiclass_boosting;
use crate::boosting::scalar_loop::sk_run_binary_boosting;

use super::classifier::SKGradientBoostingClassifier;
use super::classifier_model::SKGradientBoostingClassifierModel;
use super::importances::sk_aggregate_stage_importances;
use super::validation::{
    sk_canonicalize_targets, sk_fit_dimensions, sk_reject_streaming_mode,
    sk_validate_finite_features,
};

impl<F: SKFloat> SKSupervisedFit<F> for SKGradientBoostingClassifier<F> {
    type Model = SKGradientBoostingClassifierModel<F>;
    type Error = SKError;

    fn fit<'a, X, T>(&self, features: X, targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        sk_fit_gradient_boosting_classifier(self, features.try_into()?, targets.try_into()?)
    }
}

pub(super) fn sk_fit_gradient_boosting_classifier<F: SKFloat>(
    estimator: &SKGradientBoostingClassifier<F>,
    features: SKDataView<'_, F>,
    targets: SKTargetView<'_, F>,
) -> Result<SKGradientBoostingClassifierModel<F>, SKError> {
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
        let (class_indices, classes) = sk_canonicalize_targets(&targets)?;
        if class_indices.len() != rows_count {
            return Err(SKError::shape_mismatch_2d(
                rows_count,
                1,
                class_indices.len(),
                1,
            ));
        }
        let number_of_classes = classes.number_of_classes();
        if number_of_classes < 2 {
            return Err(SKError::InvalidHyperparameter {
                name: "targets",
                reason: "classification requires at least two classes".to_string(),
            });
        }
        let parameters = estimator.configuration.stage_parameters();
        let early_stopping = estimator.configuration.early_stopping_config();

        if number_of_classes == 2 {
            let binary_targets: Vec<F> = class_indices
                .iter()
                .map(|&class_index| {
                    if class_index == 1 {
                        F::one()
                    } else {
                        F::zero()
                    }
                })
                .collect();
            let fitted = sk_run_binary_boosting(
                &SKBinaryLogLoss::new(),
                &features,
                &class_indices,
                &binary_targets,
                estimator.number_of_estimators(),
                &parameters,
                early_stopping,
            )?;
            let feature_importances =
                sk_aggregate_stage_importances(fitted.stages.iter(), columns_count);
            let stages = fitted
                .stages
                .into_iter()
                .map(|stage| vec![stage.tree])
                .collect();
            return Ok(SKGradientBoostingClassifierModel {
                stages,
                initial_predictions: vec![fitted.initial_prediction],
                learning_rate: estimator.learning_rate(),
                classes,
                number_of_classes,
                feature_count: columns_count,
                feature_importances,
                loss: estimator.loss(),
                backend: estimator.backend(),
                execution_intent: estimator.execution_intent(),
            });
        }

        let multiclass_loss = SKMulticlassLogLoss::build()
            .number_of_classes(number_of_classes)
            .build()?;
        let fitted = sk_run_multiclass_boosting(
            &multiclass_loss,
            &features,
            &class_indices,
            estimator.number_of_estimators(),
            &parameters,
            early_stopping,
        )?;
        let feature_importances =
            sk_aggregate_stage_importances(fitted.stages.iter().flatten(), columns_count);
        let stages = fitted
            .stages
            .into_iter()
            .map(|stage_trees| stage_trees.into_iter().map(|slot| slot.tree).collect())
            .collect();
        Ok(SKGradientBoostingClassifierModel {
            stages,
            initial_predictions: fitted.initial_predictions,
            learning_rate: estimator.learning_rate(),
            classes,
            number_of_classes,
            feature_count: columns_count,
            feature_importances,
            loss: estimator.loss(),
            backend: estimator.backend(),
            execution_intent: estimator.execution_intent(),
        })
    })
}
