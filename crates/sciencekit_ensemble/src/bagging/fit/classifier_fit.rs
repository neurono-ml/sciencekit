//! Parallel per-estimator fitting for [`SKBaggingClassifier`].

use std::error::Error;

use sciencekit_common::{
    SKClassifierPredictor, SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind,
    SKSupervisedFit, SKTargetView, sk_run_operation,
};

use super::super::classifier::SKBaggingClassifier;
use super::super::error::SKBaggingError;
use super::super::model::SKBaggingClassifierModel;
use super::super::oob::sk_compute_classifier_oob_state;
use super::super::validation::{
    sk_bagging_fit_dimensions, sk_canonicalize_bagging_targets, sk_validate_bagging_finite_features,
};
use super::classifier_slot::sk_fit_classifier_slot;
use super::execution::sk_execute_estimator_slots;

/// Fit one base classifier per bootstrap sample in lock-free slots.
pub(in crate::bagging) fn sk_fit_bagging_classifier<F, B>(
    estimator: &SKBaggingClassifier<B>,
    features: SKDataView<'_, F>,
    targets: SKTargetView<'_, F>,
) -> Result<SKBaggingClassifierModel<F, B::Model>, SKBaggingError<B::Error>>
where
    F: SKFloat,
    B: SKSupervisedFit<F> + Clone + Send + Sync,
    B::Model: SKClassifierPredictor<F> + Send + Sync,
    B::Error: Error + Send + 'static + From<<B::Model as SKClassifierPredictor<F>>::Error>,
    <B::Model as SKClassifierPredictor<F>>::Error: Error + Send + 'static,
{
    let (rows_count, columns_count) = sk_bagging_fit_dimensions(&features)?;
    let attributes = SKOperationAttributes {
        operation: SKOperationKind::Fit,
        rows_count,
        columns_count,
        execution_mode: estimator.execution_intent(),
        backend: estimator.backend(),
    };
    sk_run_operation(attributes, || {
        sk_validate_bagging_finite_features(&features)?;
        let (class_indices, classes) = sk_canonicalize_bagging_targets(&targets)?;
        if class_indices.len() != rows_count {
            return Err(SKBaggingError::Central(SKError::shape_mismatch_2d(
                rows_count,
                1,
                class_indices.len(),
                1,
            )));
        }
        let number_of_classes = classes.number_of_classes();
        if number_of_classes < 2 {
            return Err(SKBaggingError::Central(SKError::InvalidHyperparameter {
                name: "targets",
                reason: "classification requires at least two classes".to_string(),
            }));
        }
        let second_row = class_indices
            .iter()
            .position(|&class_index| class_index != class_indices[0])
            .expect("at least two classes verified");

        let configuration = &estimator.configuration;
        let sample_size = configuration.maximum_samples.resolved_count(rows_count);
        let slots = sk_execute_estimator_slots(
            configuration.number_of_estimators,
            sample_size,
            estimator.execution_intent(),
            |index| {
                sk_fit_classifier_slot(
                    estimator,
                    &features,
                    &class_indices,
                    [0usize, second_row],
                    (rows_count, columns_count),
                    index,
                )
            },
        )?;

        let mut members = Vec::with_capacity(slots.len());
        let mut feature_subsets = Vec::with_capacity(slots.len());
        let mut out_of_bag_rows = configuration
            .oob_score
            .then(|| Vec::with_capacity(slots.len()));
        slots.into_iter().for_each(|slot| {
            members.push(slot.member);
            feature_subsets.push(slot.feature_subset);
            if let (Some(target), Some(rows)) = (&mut out_of_bag_rows, slot.out_of_bag_rows) {
                target.push(rows);
            }
        });

        let oob_state = sk_compute_classifier_oob_state(
            &features,
            &class_indices,
            number_of_classes,
            &members,
            &feature_subsets,
            out_of_bag_rows.as_deref(),
        )?;

        Ok(SKBaggingClassifierModel {
            members,
            feature_subsets,
            out_of_bag_rows,
            oob_state,
            classes,
            number_of_classes,
            feature_count: columns_count,
            bootstrap: configuration.bootstrap,
            bootstrap_features: configuration.bootstrap_features,
            backend: estimator.backend(),
            execution_intent: estimator.execution_intent(),
        })
    })
}
