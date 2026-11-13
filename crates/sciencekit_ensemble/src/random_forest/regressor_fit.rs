//! Parallel and sequential per-tree fitting for the random-forest regressor.

use ndarray::{Array1, ArrayView1};
use sciencekit_common::{
    SKBuilder, SKDataView, SKError, SKExecutionMode, SKFloat, SKOperationAttributes,
    SKOperationKind, SKSupervisedFit, SKTargetView, sk_run_operation,
};
use sciencekit_tree::{SKDecisionTreeRegressor, SKDecisionTreeRegressorModel};

use crate::bootstrap::{
    sk_gather_dense_rows, sk_gather_sparse_rows, sk_out_of_bag_rows, sk_sample_indices,
    sk_tree_seed,
};

use super::execution::sk_execute_tree_slots;
use super::importances::sk_regressor_importances;
use super::input_validation::{sk_fit_dimensions, sk_validate_finite_features};
use super::oob_scoring::sk_regressor_oob_state;
use super::regressor::SKRandomForestRegressor;
use super::regressor_model::SKRandomForestRegressorModel;

struct RegressorSlot<F> {
    model: SKDecisionTreeRegressorModel<F>,
    oob_rows: Option<Vec<usize>>,
}

/// Fit one CART regression tree per bootstrap sample in lock-free slots.
pub(super) fn sk_fit_random_forest_regressor<F: SKFloat>(
    estimator: &SKRandomForestRegressor<F>,
    features: SKDataView<'_, F>,
    targets: SKTargetView<'_, F>,
) -> Result<SKRandomForestRegressorModel<F>, SKError> {
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

        let responses_view = responses.view();
        let sample_size = estimator.maximum_samples.resolved_count(rows_count);
        let slots = sk_execute_tree_slots(
            estimator.number_of_trees,
            sample_size,
            estimator.execution_intent(),
            |tree_index| {
                sk_fit_regressor_slot(
                    estimator,
                    &features,
                    &responses_view,
                    rows_count,
                    tree_index,
                )
            },
        )?;

        let mut trees = Vec::with_capacity(slots.len());
        let mut oob_per_tree = estimator.oob_score.then(|| Vec::with_capacity(slots.len()));
        slots.into_iter().for_each(|slot| {
            trees.push(slot.model);
            if let (Some(target), Some(rows)) = (&mut oob_per_tree, slot.oob_rows) {
                target.push(rows);
            }
        });

        let oob_state =
            sk_regressor_oob_state(&features, &responses_view, &trees, oob_per_tree.as_deref())?;

        let feature_importances = sk_regressor_importances(&trees, columns_count);

        Ok(SKRandomForestRegressorModel {
            trees,
            out_of_bag_rows: oob_per_tree,
            oob_state,
            feature_count: columns_count,
            feature_importances,
            criterion: estimator.criterion,
            backend: estimator.backend,
            execution_intent: estimator.execution_intent(),
        })
    })
}

fn sk_fit_regressor_slot<F: SKFloat>(
    estimator: &SKRandomForestRegressor<F>,
    features: &SKDataView<'_, F>,
    responses: &ArrayView1<'_, F>,
    rows_count: usize,
    tree_index: usize,
) -> Result<RegressorSlot<F>, SKError> {
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
    let sampled_responses = Array1::from_iter(drawn.iter().map(|&row| responses[row]));

    let mut builder = SKDecisionTreeRegressor::<F>::build()
        .criterion(estimator.criterion)
        .minimum_samples_split(estimator.minimum_samples_split)
        .minimum_samples_leaf(estimator.minimum_samples_leaf)
        .maximum_features(estimator.maximum_features.clone())
        .random_seed(sk_tree_seed(estimator.random_seed, tree_index))
        .backend(estimator.backend)
        .execution_mode(SKExecutionMode::InProcessSynchronous);
    if let Some(maximum_depth) = estimator.maximum_depth {
        builder = builder.maximum_depth(maximum_depth);
    }
    let tree_estimator = builder.build()?;

    let model = match features {
        SKDataView::Dense(dense) => {
            let gathered = sk_gather_dense_rows(dense, &drawn);
            tree_estimator.fit(gathered.view(), sampled_responses.view())?
        }
        SKDataView::Sparse(sparse) => {
            let gathered = sk_gather_sparse_rows(sparse, &drawn);
            tree_estimator.fit(gathered.view(), sampled_responses.view())?
        }
        _ => unreachable!("dimensions already validated"),
    };

    Ok(RegressorSlot { model, oob_rows })
}
