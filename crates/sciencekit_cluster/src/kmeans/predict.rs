//! Assignment prediction and distance transform on the fitted model
//! (spec `k-means`, decision 8).

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKClusterLabel, SKClustererPredictor, SKDataView, SKError, SKFloat, SKOperationAttributes,
    SKOperationKind, sk_run_operation,
};

use super::distance::{sk_squared_distances_to_centers, sk_view_dimensions};
use super::iteration::sk_nearest_labels;
use super::model::SKKMeansModel;
use super::validation::sk_validate_query_features;

impl<F: SKFloat> SKClustererPredictor<F> for SKKMeansModel<F> {
    type Error = SKError;

    fn predict_clusters<'a, X>(&self, features: X) -> Result<Array1<SKClusterLabel>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        self.assignment_labels(features)
    }
}

impl<F: SKFloat> SKKMeansModel<F> {
    /// Assigns one typed cluster variant per input row.
    pub(crate) fn assignment_labels(
        &self,
        features: SKDataView<'_, F>,
    ) -> Result<Array1<SKClusterLabel>, SKError> {
        let (rows_count, columns_count) = sk_view_dimensions(&features);
        let attributes = SKOperationAttributes {
            operation: SKOperationKind::Predict,
            rows_count,
            columns_count,
            execution_mode: self.execution_mode,
            backend: self.backend,
        };
        sk_run_operation(attributes, || {
            sk_validate_query_features(features, self.number_of_features())?;
            let distances = sk_squared_distances_to_centers(features, &self.centers.view())?;
            let nearest = sk_nearest_labels(&distances);
            Ok(Array1::from_iter(
                nearest
                    .iter()
                    .map(|cluster| SKClusterLabel::cluster(*cluster as u32)),
            ))
        })
    }

    /// Euclidean distance from every row to every center, shape `rows ×
    /// clusters`.
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
            execution_mode: self.execution_mode,
            backend: self.backend,
        };
        sk_run_operation(attributes, || {
            sk_validate_query_features(features, self.number_of_features())?;
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
