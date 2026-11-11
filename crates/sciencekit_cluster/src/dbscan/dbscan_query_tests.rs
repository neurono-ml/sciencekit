//! Companion tests for the DBSCAN configuration and region query
//! (spec `dbscan`, decisions 1 and 6).

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
