//! Parallel per-estimator fitting for [`SKBaggingRegressor`].

use std::error::Error;

use ndarray::{Array1, ArrayView1};
use sciencekit_common::{
    SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind, SKRegressorPredictor,
    SKSupervisedFit, SKTargetView, sk_run_operation,
};

use crate::bootstrap::sk_out_of_bag_rows;

use super::super::error::SKBaggingError;
use super::super::model::SKBaggingRegressorModel;
use super::super::oob::sk_compute_regressor_oob_state;
use super::super::regressor::SKBaggingRegressor;
use super::super::sampling::{
    sk_bagging_feature_indices, sk_bagging_row_indices, sk_gather_dense_submatrix,
    sk_gather_sparse_submatrix,
};
use super::super::validation::{sk_bagging_fit_dimensions, sk_validate_bagging_finite_features};
use super::execution::sk_execute_estimator_slots;

struct RegressorSlot<M> {
    model: M,
    feature_subset: Vec<usize>,
    out_of_bag_rows: Option<Vec<usize>>,
}

/// Fit one base regressor per bootstrap sample in lock-free slots.
pub(in crate::bagging) fn sk_fit_bagging_regressor<F, B>(
    estimator: &SKBaggingRegressor<B>,
    features: SKDataView<'_, F>,
    targets: SKTargetView<'_, F>,
) -> Result<SKBaggingRegressorModel<F, B::Model>, SKBaggingError<B::Error>>
where
    F: SKFloat,
    B: SKSupervisedFit<F> + Clone + Send + Sync,
    B::Model: SKRegressorPredictor<F> + Send + Sync,
    B::Error: Error + Send + 'static + From<<B::Model as SKRegressorPredictor<F>>::Error>,
    <B::Model as SKRegressorPredictor<F>>::Error: Error + Send + 'static,
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
        let responses = targets.as_continuous()?;
        if responses.len() != rows_count {
            return Err(SKBaggingError::Central(SKError::shape_mismatch_2d(
                rows_count,
                1,
                responses.len(),
                1,
            )));
        }
        if let Some(position) = responses.iter().position(|value| !value.is_finite()) {
            return Err(SKBaggingError::Central(SKError::NonFiniteValue {
                position,
            }));
        }

        let responses_view = responses.view();
        let configuration = &estimator.configuration;
        let sample_size = configuration.maximum_samples.resolved_count(rows_count);
        let slots = sk_execute_estimator_slots(
            configuration.number_of_estimators,
            sample_size,
            estimator.execution_intent(),
            |estimator_index| {
                sk_fit_regressor_slot(
                    estimator,
                    &features,
                    &responses_view,
                    rows_count,
                    columns_count,
                    estimator_index,
                )
            },
        )?;

        let mut estimators = Vec::with_capacity(slots.len());
        let mut feature_subsets = Vec::with_capacity(slots.len());
        let mut out_of_bag_rows = configuration
            .oob_score
            .then(|| Vec::with_capacity(slots.len()));
        slots.into_iter().for_each(|slot| {
            estimators.push(slot.model);
            feature_subsets.push(slot.feature_subset);
            if let (Some(target), Some(rows)) = (&mut out_of_bag_rows, slot.out_of_bag_rows) {
                target.push(rows);
            }
        });

        let oob_state = sk_compute_regressor_oob_state(
            &features,
            &responses_view,
            &estimators,
            &feature_subsets,
            out_of_bag_rows.as_deref(),
        )?;

        Ok(SKBaggingRegressorModel {
            estimators,
            feature_subsets,
            out_of_bag_rows,
            oob_state,
            feature_count: columns_count,
            backend: estimator.backend(),
            execution_intent: estimator.execution_intent(),
        })
    })
}

fn sk_fit_regressor_slot<F, B>(
    estimator: &SKBaggingRegressor<B>,
    features: &SKDataView<'_, F>,
    responses: &ArrayView1<'_, F>,
    rows_count: usize,
    columns_count: usize,
    estimator_index: usize,
) -> Result<RegressorSlot<B::Model>, SKBaggingError<B::Error>>
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
    let sampled_responses = Array1::from_iter(drawn_rows.iter().map(|&row| responses[row]));

    let model = match features {
        SKDataView::Dense(dense) => {
            let submatrix = sk_gather_dense_submatrix(dense, &drawn_rows, &feature_subset);
            estimator
                .base_estimator
                .fit(submatrix.view(), sampled_responses.view())
                .map_err(SKBaggingError::Base)?
        }
        SKDataView::Sparse(sparse) => {
            let submatrix = sk_gather_sparse_submatrix(sparse, &drawn_rows, &feature_subset);
            estimator
                .base_estimator
                .fit(submatrix.view(), sampled_responses.view())
                .map_err(SKBaggingError::Base)?
        }
        _ => unreachable!("dimensions already validated"),
    };

    Ok(RegressorSlot {
        model,
        feature_subset,
        out_of_bag_rows,
    })
}
