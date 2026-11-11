//! Companion tests for the DBSCAN configuration and region query
//! (spec `dbscan`, decisions 1 and 6).

use sciencekit_common::SKError;

use super::region_query::sk_resolve_strategy;
use super::{SKDBSCANAlgorithm, SKDBSCANMetric, SKDBSCANSearchStrategy};

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
        match sk_resolve_strategy(tree_algorithm, SKDBSCANMetric::Euclidean, false, 3).unwrap_err()
        {
            SKError::UnsupportedRepresentation {
                representation,
                suggestion,
            } => {
                assert_eq!(representation, "csr");
                assert!(suggestion.contains("BruteForce"));
            }
            other => panic!("expected UnsupportedRepresentation, found {other:?}"),
        }

        match sk_resolve_strategy(tree_algorithm, SKDBSCANMetric::Cosine, true, 3).unwrap_err() {
            SKError::UnsupportedRepresentation {
                representation,
                suggestion,
            } => {
                assert_eq!(representation, "cosine distance");
                assert!(suggestion.contains("BruteForce"));
            }
            other => panic!("expected UnsupportedRepresentation, found {other:?}"),
        }
    }
}
