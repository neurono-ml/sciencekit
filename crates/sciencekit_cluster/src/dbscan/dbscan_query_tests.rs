//! Companion tests for the DBSCAN configuration and region query
//! (spec `dbscan`, decisions 1 and 6).

use ndarray::array;
use sciencekit_common::{SKBackendKind, SKDataView, SKError, SKExecutionMode};
use sciencekit_neighbors::SKTrainingData;
use sprs::TriMat;

use super::config::SKDBSCANConfiguration;
use super::region_query::{sk_build_index, sk_radius_neighborhoods, sk_resolve_strategy};
use super::{SKDBSCANAlgorithm, SKDBSCANMetric, SKDBSCANSearchStrategy};

fn assert_unsupported(error: SKError, expected_representation: &'static str) {
    match error {
        SKError::UnsupportedRepresentation {
            representation,
            suggestion,
        } => {
            assert_eq!(representation, expected_representation);
            assert!(suggestion.contains("BruteForce"));
        }
        other => panic!("expected UnsupportedRepresentation, found {other:?}"),
    }
}

fn sample_configuration(
    eps: f64,
    metric: SKDBSCANMetric,
    algorithm: SKDBSCANAlgorithm,
) -> SKDBSCANConfiguration<f64> {
    SKDBSCANConfiguration {
        eps,
        minimum_samples: 2,
        metric,
        algorithm,
        leaf_size: 2,
        backend: SKBackendKind::Faer,
        execution_mode: SKExecutionMode::Automatic,
    }
}

#[test]
fn dbscan_query_configuration_defaults_and_metric_catalogue() {
    assert_eq!(SKDBSCANMetric::default(), SKDBSCANMetric::Euclidean);
    assert_eq!(SKDBSCANAlgorithm::default(), SKDBSCANAlgorithm::Auto);
    assert!(SKDBSCANMetric::Euclidean.is_tree_compatible());
    assert!(SKDBSCANMetric::Manhattan.is_tree_compatible());
    assert!(SKDBSCANMetric::Chebyshev.is_tree_compatible());
    assert!(!SKDBSCANMetric::Cosine.is_tree_compatible());
    assert!(SKDBSCANSearchStrategy::KdTree.is_tree());
    assert!(SKDBSCANSearchStrategy::BallTree.is_tree());
    assert!(!SKDBSCANSearchStrategy::BruteForce.is_tree());
    assert_eq!(SKDBSCANMetric::ALL.len(), 4);
    for metric in SKDBSCANMetric::ALL {
        assert_eq!(
            metric.is_tree_compatible(),
            metric.neighbors_metric().is_tree_compatible()
        );
    }
}

#[test]
fn dbscan_query_auto_resolves_kd_tree_for_low_dimensional_dense_metrics() {
    for metric in [
        SKDBSCANMetric::Euclidean,
        SKDBSCANMetric::Manhattan,
        SKDBSCANMetric::Chebyshev,
    ] {
        let strategy = sk_resolve_strategy(SKDBSCANAlgorithm::Auto, metric, true, 4).unwrap();
        assert_eq!(strategy, SKDBSCANSearchStrategy::KdTree);
        assert!(strategy.is_tree());
    }
}

#[test]
fn dbscan_query_auto_resolves_ball_tree_above_twenty_features() {
    let at_boundary =
        sk_resolve_strategy(SKDBSCANAlgorithm::Auto, SKDBSCANMetric::Euclidean, true, 20).unwrap();
    let above_boundary =
        sk_resolve_strategy(SKDBSCANAlgorithm::Auto, SKDBSCANMetric::Euclidean, true, 21).unwrap();
    assert_eq!(at_boundary, SKDBSCANSearchStrategy::KdTree);
    assert_eq!(above_boundary, SKDBSCANSearchStrategy::BallTree);
}

#[test]
fn dbscan_query_auto_falls_back_to_brute_force_for_sparse_and_cosine() {
    let sparse =
        sk_resolve_strategy(SKDBSCANAlgorithm::Auto, SKDBSCANMetric::Euclidean, false, 4).unwrap();
    let cosine =
        sk_resolve_strategy(SKDBSCANAlgorithm::Auto, SKDBSCANMetric::Cosine, true, 4).unwrap();
    let explicit_brute = sk_resolve_strategy(
        SKDBSCANAlgorithm::BruteForce,
        SKDBSCANMetric::Euclidean,
        true,
        4,
    )
    .unwrap();
    assert_eq!(sparse, SKDBSCANSearchStrategy::BruteForce);
    assert_eq!(cosine, SKDBSCANSearchStrategy::BruteForce);
    assert_eq!(explicit_brute, SKDBSCANSearchStrategy::BruteForce);
}

#[test]
fn dbscan_query_explicit_tree_rejects_sparse_and_cosine_with_guidance() {
    for tree_algorithm in [SKDBSCANAlgorithm::KdTree, SKDBSCANAlgorithm::BallTree] {
        assert_unsupported(
            sk_resolve_strategy(tree_algorithm, SKDBSCANMetric::Euclidean, false, 3).unwrap_err(),
            "csr",
        );
        assert_unsupported(
            sk_resolve_strategy(tree_algorithm, SKDBSCANMetric::Cosine, true, 3).unwrap_err(),
            "cosine distance",
        );
    }
}

fn run_neighborhoods(
    configuration: &SKDBSCANConfiguration<f64>,
    training: &SKTrainingData<f64>,
    strategy: SKDBSCANSearchStrategy,
) -> Vec<Vec<usize>> {
    let index = sk_build_index(configuration, training, strategy).unwrap();
    sk_radius_neighborhoods(&index, training, configuration.eps).unwrap()
}

#[test]
fn dbscan_query_radius_neighborhoods_include_each_point_itself() {
    let data = array![[0.0_f64, 0.0], [0.2, 0.0], [5.0, 5.0]];
    let training = SKTrainingData::from_view(SKDataView::Dense(data.view()));
    let configuration =
        sample_configuration(0.3, SKDBSCANMetric::Euclidean, SKDBSCANAlgorithm::Auto);
    for strategy in [
        SKDBSCANSearchStrategy::KdTree,
        SKDBSCANSearchStrategy::BallTree,
        SKDBSCANSearchStrategy::BruteForce,
    ] {
        let neighborhoods = run_neighborhoods(&configuration, &training, strategy);
        assert_eq!(neighborhoods, vec![vec![0, 1], vec![0, 1], vec![2]]);
        for (row, neighbors) in neighborhoods.iter().enumerate() {
            assert!(neighbors.contains(&row));
        }
    }

    let cosine_configuration =
        sample_configuration(0.25, SKDBSCANMetric::Cosine, SKDBSCANAlgorithm::BruteForce);
    let cosine_neighborhoods = run_neighborhoods(
        &cosine_configuration,
        &training,
        SKDBSCANSearchStrategy::BruteForce,
    );
    for (row, neighbors) in cosine_neighborhoods.iter().enumerate() {
        assert!(neighbors.contains(&row));
    }
}

#[test]
fn dbscan_query_tree_and_brute_force_neighborhoods_agree_across_metrics() {
    use SKDBSCANSearchStrategy::{BallTree, BruteForce, KdTree};
    let dense = array![[0.0_f64, 0.0], [0.4, 0.3], [0.0, 0.5], [3.0, 3.0]];
    let dense_training = SKTrainingData::from_view(SKDataView::Dense(dense.view()));
    let mut triplet = TriMat::new(dense.dim());
    for ((row, column), &value) in dense.indexed_iter() {
        if value != 0.0 {
            triplet.add_triplet(row, column, value);
        }
    }
    let csr = triplet.to_csr();
    let sparse_training = SKTrainingData::from_view(SKDataView::Sparse(csr.view()));

    for metric in [
        SKDBSCANMetric::Euclidean,
        SKDBSCANMetric::Manhattan,
        SKDBSCANMetric::Chebyshev,
    ] {
        let configuration = sample_configuration(0.6, metric, SKDBSCANAlgorithm::Auto);
        let expected = run_neighborhoods(&configuration, &dense_training, BruteForce);
        for strategy in [KdTree, BallTree] {
            assert_eq!(
                run_neighborhoods(&configuration, &dense_training, strategy),
                expected
            );
        }
        assert_eq!(
            run_neighborhoods(&configuration, &sparse_training, BruteForce),
            expected
        );
    }
}
