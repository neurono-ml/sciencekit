//! Parallel and sequential per-tree fitting for the random-forest classifier.

use ndarray::Array1;
use sciencekit_common::{
    SKBuilder, SKDataView, SKError, SKExecutionMode, SKFloat, SKOperationAttributes,
    SKOperationKind, SKSupervisedFit, SKTargetView, sk_run_operation,
};
use sciencekit_tree::{SKClassWeight, SKDecisionTreeClassifier};

use crate::bootstrap::{
    sk_gather_dense_rows, sk_gather_sparse_rows, sk_out_of_bag_rows, sk_sample_indices,
    sk_tree_seed,
};

use super::classifier::SKRandomForestClassifier;
use super::classifier_model::{SKClassifierTreeMember, SKRandomForestClassifierModel};
use super::execution::sk_execute_tree_slots;
use super::importances::sk_classifier_importances;
use super::input_validation::{
    sk_canonicalize_targets, sk_first_occurrence_classes, sk_fit_dimensions,
    sk_validate_finite_features,
};
use super::oob_scoring::sk_classifier_oob_state;

struct ClassifierSlot<F> {
    member: SKClassifierTreeMember<F>,
    oob_rows: Option<Vec<usize>>,
}

/// Fit one CART classification tree per bootstrap sample in lock-free slots.
pub(super) fn sk_fit_random_forest_classifier<F: SKFloat>(
    estimator: &SKRandomForestClassifier<F>,
    features: SKDataView<'_, F>,
    targets: SKTargetView<'_, F>,
) -> Result<SKRandomForestClassifierModel<F>, SKError> {
    let (rows_count, columns_count) = sk_fit_dimensions(&features)?;
    let attributes = SKOperationAttributes {
        operation: SKOperationKind::Fit,
        rows_count,
        columns_count,
        execution_mode: estimator.execution_intent(),
        backend: estimator.backend(),
    };
    sk_run_operation(attributes, || {
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
        if let SKClassWeight::Explicit(weights) = &estimator.class_weight {
            if weights.len() != number_of_classes {
                return Err(SKError::ShapeMismatch {
                    expected: vec![number_of_classes],
                    found: vec![weights.len()],
                });
            }
        }
        let sample_size = estimator.maximum_samples.resolved_count(rows_count);
        let slots = sk_execute_tree_slots(
            estimator.number_of_trees,
            sample_size,
            estimator.execution_intent(),
            |tree_index| {
                sk_fit_classifier_slot(estimator, &features, &class_indices, rows_count, tree_index)
            },
        )?;
        let (members, oob_per_tree) = sk_unpack_slots(slots, estimator.oob_score);
        let oob_state = sk_classifier_oob_state(
            &features,
            &class_indices,
            number_of_classes,
            &members,
            oob_per_tree.as_deref(),
        )?;
        let feature_importances = sk_classifier_importances(&members, columns_count);
        Ok(SKRandomForestClassifierModel {
            members,
            out_of_bag_rows: oob_per_tree,
            oob_state,
            classes,
            number_of_classes,
            feature_count: columns_count,
            feature_importances,
            criterion: estimator.criterion,
            backend: estimator.backend,
            execution_intent: estimator.execution_intent(),
        })
    })
}

fn sk_unpack_slots<F>(
    slots: Vec<ClassifierSlot<F>>,
    oob_score: bool,
) -> (Vec<SKClassifierTreeMember<F>>, Option<Vec<Vec<usize>>>) {
    let mut members = Vec::with_capacity(slots.len());
    let mut oob = oob_score.then(|| Vec::with_capacity(slots.len()));
    slots.into_iter().for_each(|slot| {
        members.push(slot.member);
        if let (Some(target), Some(rows)) = (&mut oob, slot.oob_rows) {
            target.push(rows);
        }
    });
    (members, oob)
}

fn sk_fit_classifier_slot<F: SKFloat>(
    estimator: &SKRandomForestClassifier<F>,
    features: &SKDataView<'_, F>,
    class_indices: &[usize],
    rows_count: usize,
    tree_index: usize,
) -> Result<ClassifierSlot<F>, SKError> {
    let drawn = sk_sample_indices(
        rows_count,
        &estimator.maximum_samples,
        estimator.bootstrap,
        estimator.random_seed,
        tree_index,
    );
    let oob_rows = estimator
        .oob_score
        .then(|| sk_out_of_bag_rows(rows_count, &drawn));
    let sampled_classes: Vec<usize> = drawn.iter().map(|&row| class_indices[row]).collect();
    let local_to_forest = sk_first_occurrence_classes(&sampled_classes);

    let pure_leaf = |class_index| ClassifierSlot {
        member: SKClassifierTreeMember::PureLeaf { class_index },
        oob_rows: oob_rows.clone(),
    };
    let local_weight = match &estimator.class_weight {
        SKClassWeight::None => SKClassWeight::None,
        SKClassWeight::Balanced => SKClassWeight::Balanced,
        SKClassWeight::Explicit(weights) => {
            let mapped: Vec<F> = local_to_forest
                .iter()
                .map(|&class_index| weights[class_index])
                .collect();
            if mapped.iter().all(|weight| *weight == F::zero()) {
                return Ok(pure_leaf(local_to_forest[0]));
            }
            SKClassWeight::Explicit(mapped)
        }
    };
    if local_to_forest.len() < 2 {
        return Ok(pure_leaf(local_to_forest[0]));
    }

    let mut builder = SKDecisionTreeClassifier::<F>::build()
        .criterion(estimator.criterion)
        .minimum_samples_split(estimator.minimum_samples_split)
        .minimum_samples_leaf(estimator.minimum_samples_leaf)
        .maximum_features(estimator.maximum_features.clone())
        .random_seed(sk_tree_seed(estimator.random_seed, tree_index))
        .class_weight(local_weight)
        .backend(estimator.backend)
        .execution_mode(SKExecutionMode::InProcessSynchronous);
    if let Some(maximum_depth) = estimator.maximum_depth {
        builder = builder.maximum_depth(maximum_depth);
    }
    let tree_estimator = builder.build()?;
    let sampled_labels = Array1::from_iter(
        sampled_classes
            .iter()
            .map(|&class_index| class_index as i64),
    );
    let model = match features {
        SKDataView::Dense(dense) => tree_estimator.fit(
            sk_gather_dense_rows(dense, &drawn).view(),
            sampled_labels.view(),
        )?,
        SKDataView::Sparse(sparse) => tree_estimator.fit(
            sk_gather_sparse_rows(sparse, &drawn).view(),
            sampled_labels.view(),
        )?,
        _ => unreachable!("dimensions already validated"),
    };
    Ok(ClassifierSlot {
        member: SKClassifierTreeMember::Fitted {
            model,
            local_to_forest,
        },
        oob_rows,
    })
}
