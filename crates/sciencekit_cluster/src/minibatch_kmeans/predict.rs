//! Nearest-center assignment (`SKClustererPredictor`, `predict`) and Euclidean
//! distance `transform` on [`SKMiniBatchKMeansModel`] (spec
//! `mini-batch-k-means`).

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKClusterLabel, SKClustererPredictor, SKDataView, SKError, SKFloat, SKOperationAttributes,
    SKOperationKind, sk_run_operation,
};

use super::assign::sk_assign_batch_rows;
use super::model::SKMiniBatchKMeansModel;
use super::validation::sk_validate_query_features;
use crate::kmeans::distance::{sk_squared_distances_to_centers, sk_view_dimensions};

impl<F: SKFloat> SKClustererPredictor<F> for SKMiniBatchKMeansModel<F> {
    type Error = SKError;

    fn predict_clusters<'a, X>(&self, features: X) -> Result<Array1<SKClusterLabel>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        let (rows_count, columns_count) = sk_view_dimensions(&features);
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Predict,
            rows_count,
            columns_count,
            execution_mode: self.configuration.execution_mode,
            backend: self.configuration.backend,
        };
        sk_run_operation(attributes, || {
            sk_validate_query_features(features, self.number_of_features(), "predict")?;
            let (nearest, _) = sk_assign_batch_rows(features, &self.centers.view())?;
            Ok(Array1::from_iter(
                nearest
                    .iter()
                    .map(|&cluster| SKClusterLabel::cluster(cluster as u32)),
            ))
        })
    }
}

impl<F: SKFloat> SKMiniBatchKMeansModel<F> {
    /// Assigns every query row to its nearest cluster center.
    pub fn predict<'a, X>(&self, features: X) -> Result<Array1<SKClusterLabel>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        self.predict_clusters(features)
    }

    /// Euclidean distances from every query row to every cluster center, shape
    /// `(number_of_rows, number_of_clusters)`.
    pub fn transform<'a, X>(&self, features: X) -> Result<Array2<F>, SKError>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        let (rows_count, columns_count) = sk_view_dimensions(&features);
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Transform,
            rows_count,
            columns_count,
            execution_mode: self.configuration.execution_mode,
            backend: self.configuration.backend,
        };
        sk_run_operation(attributes, || {
            sk_validate_query_features(features, self.number_of_features(), "transform")?;
            let squared = sk_squared_distances_to_centers(features, &self.centers.view())?;
            Ok(squared.mapv(|distance| {
                if distance > F::zero() {
                    distance.sqrt()
                } else {
                    F::zero()
                }
            }))
        })
    }
}
