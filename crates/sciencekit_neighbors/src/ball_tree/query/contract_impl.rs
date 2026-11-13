//! The shared searcher contract implementation for the ball tree.

use sciencekit_common::{SKDataView, SKError, SKFloat};

use super::super::build::SKBallTree;
use crate::searcher::candidates::sk_should_run_parallel;
use crate::searcher::{
    SKNearestNeighborsSearcher, SKNeighborQuery, SKRadiusNeighbors, sk_require_dense,
    sk_validate_query, sk_validate_radius,
};

impl<F: SKFloat> SKNearestNeighborsSearcher<F> for SKBallTree<F> {
    fn k_nearest(
        &self,
        queries: SKDataView<'_, F>,
        neighbors_count: usize,
    ) -> Result<SKNeighborQuery<F>, SKError> {
        let dense = sk_require_dense(queries)?;
        sk_validate_query(
            dense,
            self.training_features(),
            self.training_rows(),
            neighbors_count,
        )?;
        let span = tracing::info_span!(
            "sciencekit.neighbors.k_nearest",
            operation = "k_nearest",
            backend = %self.backend,
            rows = dense.nrows(),
            columns = dense.ncols(),
        );
        let _entered = span.enter();
        let parallel = sk_should_run_parallel(dense.nrows(), self.training_rows());
        Ok(self.k_nearest_with_mode(dense, neighbors_count, parallel))
    }

    fn radius_neighbors(
        &self,
        queries: SKDataView<'_, F>,
        radius: F,
    ) -> Result<SKRadiusNeighbors<F>, SKError> {
        let dense = sk_require_dense(queries)?;
        sk_validate_radius(dense, self.training_features(), radius)?;
        let span = tracing::info_span!(
            "sciencekit.neighbors.radius_neighbors",
            operation = "radius_neighbors",
            backend = %self.backend,
            rows = dense.nrows(),
            columns = dense.ncols(),
        );
        let _entered = span.enter();
        let parallel = sk_should_run_parallel(dense.nrows(), self.training_rows());
        Ok(self.radius_neighbors_with_mode(dense, radius, parallel))
    }
}
