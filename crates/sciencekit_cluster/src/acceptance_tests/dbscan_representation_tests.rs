//! Representation and search-strategy parity acceptance tests for `SKDBSCAN`.

use ndarray::Array2;
use sciencekit_common::{SKBuilder, SKClustererPredictor, SKUnsupervisedFit};
use sprs::CsMat;

use super::dbscan_fixtures::planted_density_chains;
use super::fixtures::csr_from_dense;
use crate::{SKDBSCAN, SKDBSCANAlgorithm, SKDBSCANSearchStrategy};

#[test]
fn dbscan_acceptance_representation_dense_sparse_and_trees_agree() {
    let (dense, _) = planted_density_chains(600, 3, 10, 25);
    let sparse = csr_from_dense(&dense);

    let auto_dense = SKDBSCAN::<f64>::build()
        .eps(0.45)
        .minimum_samples(5)
        .build()
        .unwrap()
        .fit(&dense)
        .unwrap();
    let ball_dense = SKDBSCAN::<f64>::build()
        .eps(0.45)
        .minimum_samples(5)
        .algorithm(SKDBSCANAlgorithm::BallTree)
        .build()
        .unwrap()
        .fit(&dense)
        .unwrap();
    let brute_dense = SKDBSCAN::<f64>::build()
        .eps(0.45)
        .minimum_samples(5)
        .algorithm(SKDBSCANAlgorithm::BruteForce)
        .build()
        .unwrap()
        .fit(&dense)
        .unwrap();
    let auto_sparse = SKDBSCAN::<f64>::build()
        .eps(0.45)
        .minimum_samples(5)
        .build()
        .unwrap()
        .fit(sparse.view())
        .unwrap();

    assert_eq!(auto_dense.search_strategy(), SKDBSCANSearchStrategy::KdTree);
    assert_eq!(
        ball_dense.search_strategy(),
        SKDBSCANSearchStrategy::BallTree
    );
    assert_eq!(
        brute_dense.search_strategy(),
        SKDBSCANSearchStrategy::BruteForce
    );
    assert_eq!(
        auto_sparse.search_strategy(),
        SKDBSCANSearchStrategy::BruteForce
    );
    assert_eq!(auto_sparse.training_representation(), "csr");

    assert_eq!(auto_dense.labels(), ball_dense.labels());
    assert_eq!(auto_dense.labels(), brute_dense.labels());
    assert_eq!(auto_dense.labels(), auto_sparse.labels());
    assert_eq!(
        auto_dense.core_sample_indices(),
        auto_sparse.core_sample_indices()
    );
    assert_eq!(
        auto_dense.predict_clusters(sparse.view()).unwrap(),
        auto_sparse.predict_clusters(&dense).unwrap()
    );
}

#[test]
fn dbscan_acceptance_representation_high_dimensional_auto_uses_ball_tree_and_wide_csr_stays_sparse()
{
    let high_dimensional = Array2::from_shape_fn((30, 24), |(row, column)| {
        let group = (row / 10) as f64 * 50.0;
        group + ((row + column) % 3) as f64 * 0.02
    });
    let high_dimensional_model = SKDBSCAN::<f64>::build()
        .eps(1.0)
        .minimum_samples(4)
        .build()
        .unwrap()
        .fit(&high_dimensional)
        .unwrap();
    assert_eq!(
        high_dimensional_model.search_strategy(),
        SKDBSCANSearchStrategy::BallTree
    );
    assert_eq!(high_dimensional_model.number_of_clusters(), 3);

    let width = 5_000_000;
    let csr = CsMat::new(
        (4, width),
        vec![0, 1, 2, 3, 4],
        vec![0, 0, 0, width - 1],
        vec![1.0_f64, 1.05, 1.1, 20.0],
    );
    let wide_model = SKDBSCAN::<f64>::build()
        .eps(0.2)
        .minimum_samples(3)
        .build()
        .unwrap()
        .fit(csr.view())
        .unwrap();
    assert_eq!(wide_model.training_representation(), "csr");
    assert_eq!(wide_model.number_of_clusters(), 1);
    assert_eq!(wide_model.number_of_features(), width);
}
