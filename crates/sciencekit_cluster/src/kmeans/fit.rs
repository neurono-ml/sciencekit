//! The `SKUnsupervisedFit` implementation (spec `k-means`, decision 5).

use ndarray::Array1;
use sciencekit_common::{
    SKClusterLabel, SKDataView, SKError, SKFloat, SKOperationAttributes, SKOperationKind,
    SKUnsupervisedFit, sk_run_operation,
};

use super::algorithm::sk_fit_with_algorithm;
use super::core_implementation::SKKMeans;
use super::distance::sk_view_dimensions;
use super::init::{sk_initialize_centers, sk_run_initializations};
use super::model::SKKMeansModel;
use super::validation::sk_validate_fit_features;

impl<F: SKFloat> SKUnsupervisedFit<F> for SKKMeans<F> {
    type Model = SKKMeansModel<F>;
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

impl<F: SKFloat> SKKMeans<F> {
    /// Fits the estimator and returns the training assignments in one call
    /// (spec `k-means`, decision 8).
    pub fn fit_predict<'a, X>(&self, features: X) -> Result<Array1<SKClusterLabel>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let model = self.fit(features)?;
        Ok(model.labels.to_owned())
    }
}

/// Validates the input, runs every initialization and keeps the lowest
/// inertia.
fn sk_fit_estimator<F: SKFloat>(
    estimator: &SKKMeans<F>,
    features: SKDataView<'_, F>,
) -> Result<SKKMeansModel<F>, SKError> {
    let configuration = estimator.configuration;
    sk_validate_fit_features(features, configuration.number_of_clusters)?;

    type InitializationResult<F> = (super::result::SKKMeansFitResult<F>, super::SKKMeansAlgorithm);
    let (best, _) = sk_run_initializations::<InitializationResult<F>, SKError>(
        configuration.number_of_initializations,
        configuration.random_seed,
        |_, seed| {
            let centers = sk_initialize_centers(
                features,
                configuration.number_of_clusters,
                configuration.initialization,
                seed,
            )?;
            let (result, algorithm) = sk_fit_with_algorithm(
                features,
                centers,
                configuration.algorithm,
                configuration.maximum_number_of_iterations,
                configuration.tolerance,
            )?;
            let inertia = result.inertia.to_f64().expect("a finite scalar converts");
            Ok(((result, algorithm), inertia))
        },
    )?;

    let (result, algorithm) = best;
    let labels = Array1::from_iter(
        result
            .labels
            .iter()
            .map(|cluster| SKClusterLabel::cluster(*cluster as u32)),
    );
    Ok(SKKMeansModel::from_fit(
        result.centers,
        labels,
        result.inertia,
        result.iterations,
        result.converged,
        algorithm,
        configuration.backend,
        configuration.execution_mode,
    ))
}
