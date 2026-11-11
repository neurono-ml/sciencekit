//! Fitting logic for `SKIsolationForest` (spec `isolation-forest`).

use sciencekit_common::{
    SKDataView, SKError, SKExecutionMode, SKFloat, SKOperationAttributes, SKOperationKind,
    SKUnsupervisedFit, sk_run_operation,
};
use sciencekit_tree::SKTree;

use super::config::{SK_ISOLATION_AUTO_OFFSET, SKContamination};
use super::core_implementation::SKIsolationForest;
use super::execution::sk_execute_tree_slots;
use super::model::SKIsolationForestModel;
use super::score::{sk_compute_anomaly_scores, sk_exact_quantile};
use super::sparse::sk_gather_dense_rows;
use super::subsample::{sk_prepare_tree_draws, sk_resolve_feature_count};
use super::tree::{
    sk_build_dense_isolation_tree, sk_build_sparse_isolation_tree, sk_isolation_depth_limit,
};
use super::validation::{sk_validate_fit_features, sk_view_dimensions};

impl<F: SKFloat> SKUnsupervisedFit<F> for SKIsolationForest<F> {
    type Model = SKIsolationForestModel<F>;
    type Error = SKError;

    fn fit<'a, X>(&self, features: X) -> Result<Self::Model, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        let (rows_count, columns_count) = sk_view_dimensions(&features);
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Fit,
            rows_count,
            columns_count,
            execution_mode: self.execution_mode(),
            backend: self.backend,
        };
        sk_run_operation(attributes, || {
            sk_fit_isolation_forest(self, &features, "fit")
        })
    }
}

impl<F: SKFloat> SKIsolationForest<F> {
    /// Fits the Isolation Forest on `features` and returns the learned model.
    pub fn fit<'a, X>(&self, features: X) -> Result<SKIsolationForestModel<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        <Self as SKUnsupervisedFit<F>>::fit(self, features)
    }
}

/// Internal fit routine reusable by both `fit` and `fit_predict` inside a single span.
pub(super) fn sk_fit_isolation_forest<F: SKFloat>(
    estimator: &SKIsolationForest<F>,
    features: &SKDataView<'_, F>,
    operation: &'static str,
) -> Result<SKIsolationForestModel<F>, SKError> {
    let (rows_count, columns_count) =
        sk_validate_fit_features(features, estimator.maximum_samples, operation)?;
    let subsample_size = estimator.maximum_samples.resolved_count(rows_count);
    let depth_limit = sk_isolation_depth_limit(subsample_size);
    let feature_draw_count = sk_resolve_feature_count(estimator.maximum_features, columns_count);

    let trees = match features {
        SKDataView::Dense(dense) => sk_execute_tree_slots(
            estimator.number_of_trees,
            subsample_size,
            estimator.execution_mode(),
            |tree_index| {
                let (sampled_rows, candidate_features, mut split_generator) = sk_prepare_tree_draws(
                    estimator.random_seed,
                    tree_index,
                    rows_count,
                    subsample_size,
                    estimator.bootstrap,
                    columns_count,
                    feature_draw_count,
                );
                let gathered = sk_gather_dense_rows(*dense, &sampled_rows);
                Ok(sk_build_dense_isolation_tree(
                    gathered.view(),
                    &candidate_features,
                    depth_limit,
                    &mut split_generator,
                ))
            },
        )?,
        SKDataView::Sparse(sparse) => sk_execute_tree_slots(
            estimator.number_of_trees,
            subsample_size,
            estimator.execution_mode(),
            |tree_index| {
                let (sampled_rows, candidate_features, mut split_generator) = sk_prepare_tree_draws(
                    estimator.random_seed,
                    tree_index,
                    rows_count,
                    subsample_size,
                    estimator.bootstrap,
                    columns_count,
                    feature_draw_count,
                );
                Ok(sk_build_sparse_isolation_tree(
                    *sparse,
                    &sampled_rows,
                    &candidate_features,
                    depth_limit,
                    &mut split_generator,
                ))
            },
        )?,
        _ => unreachable!("fit view already validated"),
    };

    let offset = compute_training_offset(
        &trees,
        subsample_size,
        features,
        estimator.contamination,
        estimator.execution_mode(),
    )?;

    Ok(SKIsolationForestModel {
        trees,
        subsample_size,
        feature_count: columns_count,
        offset,
        backend: estimator.backend,
        execution_mode: estimator.execution_mode(),
    })
}

fn compute_training_offset<F: SKFloat>(
    trees: &[SKTree<F>],
    subsample_size: usize,
    features: &SKDataView<'_, F>,
    contamination: SKContamination<F>,
    execution_mode: SKExecutionMode,
) -> Result<F, SKError> {
    match contamination {
        SKContamination::Auto => Ok(F::from(SK_ISOLATION_AUTO_OFFSET).expect("-0.5 fits scalar")),
        SKContamination::Fraction(fraction) => {
            let anomalies =
                sk_compute_anomaly_scores(trees, subsample_size, features, execution_mode)?;
            let mut score_samples: Vec<F> = anomalies.into_iter().map(|score| -score).collect();
            score_samples.sort_by(|left, right| {
                left.partial_cmp(right)
                    .unwrap_or(core::cmp::Ordering::Equal)
            });
            Ok(sk_exact_quantile(&score_samples, fraction))
        }
    }
}
