//! The `SKUnsupervisedFit` and `fit_predict` entry points for DBSCAN
//! (spec `dbscan`).

use ndarray::Array1;
use sciencekit_common::{
    SKClusterLabel, SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind,
    SKUnsupervisedFit, sk_run_operation,
};
use sciencekit_neighbors::{SKBruteForceSearcher, SKTrainingData};

use super::core_implementation::SKDBSCAN;
use super::expansion::{sk_extract_core_samples, sk_run_dbscan_partition};
use super::model::{SKDBSCANCoreSearch, SKDBSCANModel};
use super::region_query::sk_build_index;
use super::validation::{sk_validate_fit_features, sk_view_dimensions};

impl<F: SKFloat> SKUnsupervisedFit<F> for SKDBSCAN<F> {
    type Model = SKDBSCANModel<F>;
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
            execution_mode: self.configuration.execution_mode,
            backend: self.configuration.backend,
        };
        sk_run_operation(attributes, || sk_fit_estimator(self, features))
    }
}

impl<F: SKFloat> SKDBSCAN<F> {
    /// Fits the estimator and returns the typed training assignments in one
    /// call.
    pub fn fit_predict<'a, X>(&self, features: X) -> Result<Array1<SKClusterLabel>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        let (rows_count, columns_count) = sk_view_dimensions(&features);
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::FitPredict,
            rows_count,
            columns_count,
            execution_mode: self.configuration.execution_mode,
            backend: self.configuration.backend,
        };
        sk_run_operation(attributes, || {
            let model = sk_fit_estimator(self, features)?;
            Ok(model.labels)
        })
    }
}

/// Validate the input, run the DBSCAN partition and construct the model.
fn sk_fit_estimator<F: SKFloat>(
    estimator: &SKDBSCAN<F>,
    features: SKDataView<'_, F>,
) -> Result<SKDBSCANModel<F>, SKError> {
    let configuration = estimator.configuration;
    sk_validate_fit_features(features)?;

    let training = SKTrainingData::from_view(features);
    let expansion = sk_run_dbscan_partition(&configuration, &training)?;
    let core_search = if expansion.core_sample_indices.is_empty() {
        None
    } else {
        let core_indices = expansion
            .core_sample_indices
            .as_slice()
            .expect("core_sample_indices is contiguous");
        let core_samples = sk_extract_core_samples(&training, core_indices);
        let labels = core_indices
            .iter()
            .map(|&index| expansion.labels[index])
            .collect();
        let tree = if expansion.strategy.is_tree() {
            Some(sk_build_index(
                &configuration,
                &core_samples,
                expansion.strategy,
            )?)
        } else {
            None
        };
        let brute_force =
            SKBruteForceSearcher::build(core_samples, configuration.metric.neighbors_metric())?;
        Some(SKDBSCANCoreSearch {
            labels,
            tree,
            brute_force,
        })
    };

    Ok(SKDBSCANModel::from_expansion(
        expansion,
        training.columns(),
        training.representation(),
        core_search,
        configuration,
    ))
}
