//! Companion tests for the shared searcher contract.

use ndarray::Array2;
use sciencekit_common::{SKDataView, SKError};

use super::contract::SKNearestNeighborsSearcher;
use super::{SKNeighborQuery, SKRadiusNeighbors};

/// A contract-only searcher proving the trait can be used behind a trait
/// object by the estimator's runtime algorithm enum.
struct SKMockSearcher;

impl SKNearestNeighborsSearcher<f64> for SKMockSearcher {
    fn k_nearest(
        &self,
        queries: SKDataView<'_, f64>,
        neighbors_count: usize,
    ) -> Result<SKNeighborQuery<f64>, SKError> {
        let rows = queries.as_dense()?.nrows();
        Ok(SKNeighborQuery::new(
            Array2::zeros((rows, neighbors_count)),
            Array2::zeros((rows, neighbors_count)),
        ))
    }

    fn radius_neighbors(
        &self,
        queries: SKDataView<'_, f64>,
        _radius: f64,
    ) -> Result<SKRadiusNeighbors<f64>, SKError> {
        let rows = queries.as_dense()?.nrows();
        Ok(SKRadiusNeighbors::new(vec![vec![]; rows], vec![vec![]; rows]))
    }
}

#[test]
fn searcher_contract_is_usable_behind_a_trait_object() {
    let searcher: Box<dyn SKNearestNeighborsSearcher<f64>> = Box::new(SKMockSearcher);
    let queries = Array2::<f64>::zeros((3, 2));

    let neighbors = searcher
        .k_nearest(SKDataView::Dense(queries.view()), 4)
        .expect("the mock never fails");
    assert_eq!(neighbors.distances().shape(), &[3, 4]);
    assert_eq!(neighbors.indices().shape(), &[3, 4]);

    let radius = searcher
        .radius_neighbors(SKDataView::Dense(queries.view()), 1.0)
        .expect("the mock never fails");
    assert_eq!(radius.rows(), 3);
}
