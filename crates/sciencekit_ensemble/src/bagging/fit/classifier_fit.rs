//! Parallel per-estimator fitting for [`SKBaggingClassifier`].

use std::error::Error;

use ndarray::Array1;
use sciencekit_common::{
    SKClassifierPredictor, SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind,
    SKSupervisedFit, SKTargetView, sk_run_operation,
};

use crate::bootstrap::sk_out_of_bag_rows;

use super::super::classifier::SKBaggingClassifier;
use super::super::error::SKBaggingError;
use super::super::model::{SKBaggingClassifierMember, SKBaggingClassifierModel, SKBaggingOobState};
use super::super::sampling::{
    sk_bagging_feature_indices, sk_bagging_row_indices, sk_gather_dense_submatrix,
    sk_gather_sparse_submatrix,
};
use super::super::validation::{
    sk_bagging_fit_dimensions, sk_canonicalize_bagging_targets, sk_first_occurrence_classes,
    sk_validate_bagging_finite_features,
};
use super::execution::sk_execute_estimator_slots;

struct ClassifierSlot<M> {
    member: SKBaggingClassifierMember<M>,
    feature_subset: Vec<usize>,
    out_of_bag_rows: Option<Vec<usize>>,
}

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
    B::Error: Error + Send + 'static,
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
            |estimator_index| {
                sk_fit_classifier_slot(
                    estimator,
                    &features,
                    &class_indices,
                    [0usize, second_row],
                    (rows_count, columns_count),
                    estimator_index,
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

        Ok(SKBaggingClassifierModel {
            members,
            feature_subsets,
            out_of_bag_rows,
            oob_state: SKBaggingOobState::Disabled,
            classes,
            number_of_classes,
            feature_count: columns_count,
            backend: estimator.backend(),
            execution_intent: estimator.execution_intent(),
        })
    })
}

fn sk_fit_classifier_slot<F, B>(
    estimator: &SKBaggingClassifier<B>,
    features: &SKDataView<'_, F>,
    class_indices: &[usize],
    two_class_pair: [usize; 2],
    (rows_count, columns_count): (usize, usize),
    estimator_index: usize,
) -> Result<ClassifierSlot<B::Model>, SKBaggingError<B::Error>>
where
    F: SKFloat,
    B: SKSupervisedFit<F> + Clone + Send + Sync,
{
    let configuration = &estimator.configuration;
    let drawn_rows = sk_bagging_row_indices(
        rows_count,
        &configuration.maximum_samples,
        configuration.bootstrap,
        configuration.random_seed,
        estimator_index,
    );
    let feature_subset = sk_bagging_feature_indices(
        columns_count,
        &configuration.maximum_features,
        configuration.bootstrap_features,
        configuration.random_seed,
        estimator_index,
    );
    let out_of_bag_rows = configuration
        .oob_score
        .then(|| sk_out_of_bag_rows(rows_count, &drawn_rows));
    let sampled_classes: Vec<usize> = drawn_rows.iter().map(|&row| class_indices[row]).collect();
    let local_to_ensemble = sk_first_occurrence_classes(&sampled_classes);
    let sampled_labels = Array1::from_iter(
        sampled_classes
            .iter()
            .map(|&class_index| class_index as i64),
    );

    let member = match sk_fit_base_on_subset(
        &estimator.base_estimator,
        features,
        &drawn_rows,
        &feature_subset,
        &sampled_labels,
    ) {
        Ok(model) => SKBaggingClassifierMember::Fitted {
            model,
            local_to_ensemble,
        },
        Err(_) if local_to_ensemble.len() < 2 => {
            let probe_labels = Array1::from_vec(vec![0_i64, 1]);
            sk_fit_base_on_subset(
                &estimator.base_estimator,
                features,
                &two_class_pair,
                &feature_subset,
                &probe_labels,
            )
            .map_err(SKBaggingError::Base)?;
            SKBaggingClassifierMember::ConstantClass {
                class_index: local_to_ensemble[0],
            }
        }
        Err(base_error) => return Err(SKBaggingError::Base(base_error)),
    };

    Ok(ClassifierSlot {
        member,
        feature_subset,
        out_of_bag_rows,
    })
}

fn sk_fit_base_on_subset<F, B>(
    base_estimator: &B,
    features: &SKDataView<'_, F>,
    rows: &[usize],
    feature_subset: &[usize],
    labels: &Array1<i64>,
) -> Result<B::Model, B::Error>
where
    F: SKFloat,
    B: SKSupervisedFit<F>,
{
    match features {
        SKDataView::Dense(dense) => base_estimator.fit(
            sk_gather_dense_submatrix(dense, rows, feature_subset).view(),
            labels.view(),
        ),
        SKDataView::Sparse(sparse) => base_estimator.fit(
            sk_gather_sparse_submatrix(sparse, rows, feature_subset).view(),
            labels.view(),
        ),
        _ => unreachable!("dimensions already validated"),
    }
}
