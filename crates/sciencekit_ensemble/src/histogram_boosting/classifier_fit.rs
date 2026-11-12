//! Fitting entry point for [`SKHistGradientBoostingClassifier`].

use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKSupervisedFit,
    SKTargetView, sk_run_operation,
};

use crate::boosting::loss::SKBinaryLogLoss;
use crate::histogram_boosting::binning::SKBinnedDataset;
use crate::histogram_boosting::fit::{
    sk_aggregate_histogram_importances, sk_validation_rows,
};
use crate::histogram_boosting::multiclass_loop::sk_run_histogram_multiclass;
use crate::histogram_boosting::scalar_loop::sk_run_histogram_scalar;

use super::classifier::SKHistGradientBoostingClassifier;
use super::classifier_model::SKHistGradientBoostingClassifierModel;
use super::validation::{
    sk_canonicalize_targets, sk_fit_dimensions, sk_reject_streaming_mode,
    sk_validate_finite_or_missing_features,
};

impl<F: SKFloat> SKSupervisedFit<F> for SKHistGradientBoostingClassifier<F> {
    type Model = SKHistGradientBoostingClassifierModel<F>;
    type Error = SKError;

    fn fit<'a, X, T>(&self, features: X, targets: T) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
        T: TryInto<SKTargetView<'a, F>, Error = SKError>,
    {
        sk_fit_histogram_classifier(self, features.try_into()?, targets.try_into()?)
    }
}

pub(super) fn sk_fit_histogram_classifier<F: SKFloat>(
    estimator: &SKHistGradientBoostingClassifier<F>,
    features: SKDataView<'_, F>,
    targets: SKTargetView<'_, F>,
) -> Result<SKHistGradientBoostingClassifierModel<F>, SKError> {
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
        let dataset = SKBinnedDataset::build(
            &features,
            estimator.categorical_features(),
            estimator.maximum_bins(),
        )?;
        let parameters = estimator.configuration.stage_parameters();
        let early_stopping = estimator.configuration.early_stopping_config();
        let (training_rows, validation_rows) = sk_validation_rows(
            rows_count,
            Some(&class_indices),
            number_of_classes,
            early_stopping,
        );
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
            let fitted = sk_run_histogram_scalar(
                &SKBinaryLogLoss::new(),
                &dataset,
                &binary_targets,
                &training_rows,
                validation_rows.as_deref(),
                estimator.number_of_estimators(),
                &parameters,
                early_stopping,
            )?;
            let tree_refs: Vec<_> = fitted.stages.iter().collect();
            let feature_importances = sk_aggregate_histogram_importances(
                &fitted.stage_importances,
                &tree_refs,
                columns_count,
            );
            let stages = fitted.stages.into_iter().map(|tree| vec![tree]).collect();
            let stage_importances = fitted
                .stage_importances
                .into_iter()
                .map(|importances| vec![importances])
                .collect();
            return Ok(SKHistGradientBoostingClassifierModel {
                stages,
                stage_importances,
                initial_predictions: vec![fitted.initial_prediction],
                learning_rate: estimator.learning_rate(),
                classes,
                number_of_classes,
                feature_count: columns_count,
                feature_importances,
                feature_binnings: dataset.feature_binnings().to_vec(),
                backend: estimator.backend(),
                execution_intent: estimator.execution_intent(),
            });
        }
        let multiclass_loss = crate::boosting::loss::SKMulticlassLogLoss::build()
            .number_of_classes(number_of_classes)
            .build()?;
        let fitted = sk_run_histogram_multiclass(
            &multiclass_loss,
            &dataset,
            &class_indices,
            &training_rows,
            validation_rows.as_deref(),
            estimator.number_of_estimators(),
            &parameters,
            early_stopping,
        )?;
        let flattened_trees = fitted.stages.iter().flatten().collect::<Vec<_>>();
        let flattened_importances = fitted
            .stage_importances
            .iter()
            .flatten()
            .cloned()
            .collect::<Vec<_>>();
        let feature_importances = sk_aggregate_histogram_importances(
            &flattened_importances,
            &flattened_trees,
            columns_count,
        );
        Ok(SKHistGradientBoostingClassifierModel {
            stages: fitted.stages,
            stage_importances: fitted.stage_importances,
            initial_predictions: fitted.initial_predictions,
            learning_rate: estimator.learning_rate(),
            classes,
            number_of_classes,
            feature_count: columns_count,
            feature_importances,
            feature_binnings: dataset.feature_binnings().to_vec(),
            backend: estimator.backend(),
            execution_intent: estimator.execution_intent(),
        })
    })
}
