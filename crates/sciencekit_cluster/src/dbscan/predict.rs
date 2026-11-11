//! Nearest-core cluster assignment for new query rows (spec `dbscan`,
//! decision 4).
//!
//! Standard DBSCAN has no out-of-sample `predict` because adding new points can
//! in principle merge density components. `SKDBSCANModel` implements
//! [`SKClustererPredictor`] as a documented **nearest-core heuristic**: each
//! query row is assigned to the cluster of its nearest core sample when that
//! distance is at most `eps`, and to [`SKClusterLabel::Noise`] otherwise.

use ndarray::Array1;
use sciencekit_common::{
    SKClusterLabel, SKClustererPredictor, SKDataView, SKError, SKFloat, SKOperationAttributes,
    SKOperationKind, sk_run_operation,
};
use sciencekit_neighbors::SKNearestNeighborsSearcher;

use super::model::SKDBSCANModel;
use super::validation::{sk_validate_query_features, sk_view_dimensions};

impl<F: SKFloat> SKClustererPredictor<F> for SKDBSCANModel<F> {
    type Error = SKError;

    fn predict_clusters<'a, X>(&self, features: X) -> Result<Array1<SKClusterLabel>, Self::Error>
    where
        X: TryInto<SKDataView<'a, F>, Error = SKError>,
    {
        let features = features.try_into()?;
        self.assignment_labels(features)
    }
}

impl<F: SKFloat> SKDBSCANModel<F> {
    /// Assigns one typed label (`Cluster` or `Noise`) per query row via the
    /// nearest fitted core sample within the inclusive `eps` radius.
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
            sk_validate_query_features(features, self.number_of_features)?;
            let Some(core_search) = &self.core_search else {
                return Ok(Array1::from_elem(rows_count, SKClusterLabel::Noise));
            };

            let nearest = match (&core_search.tree, features) {
                (Some(tree), SKDataView::Dense(_)) => tree.k_nearest(features, 1)?,
                _ => core_search.brute_force.k_nearest(features, 1)?,
            };

            Ok(Array1::from_iter((0..rows_count).map(|row| {
                let distance = nearest.distances()[[row, 0]];
                let core_position = nearest.indices()[[row, 0]];
                if distance <= self.eps {
                    core_search.labels[core_position]
                } else {
                    SKClusterLabel::Noise
                }
            })))
        })
    }
}
