//! Companion tests for the runtime search-strategy resolution
//! (spec `nearest-neighbor-estimators`).

use sciencekit_common::SKError;

use super::{
    SK_AUTO_BALL_TREE_MINIMUM_FEATURES, SK_AUTO_TREE_MINIMUM_TRAINING_ROWS,
    SKFeatureRepresentation, SKNeighborsAlgorithm, SKNeighborsMetric, SKSearchStrategy,
    sk_resolve_search_strategy,
};

fn resolve(
    algorithm: SKNeighborsAlgorithm,
    metric: SKNeighborsMetric,
    representation: SKFeatureRepresentation,
    rows: usize,
    columns: usize,
) -> Result<SKSearchStrategy, SKError> {
    sk_resolve_search_strategy(algorithm, metric, representation, rows, columns)
}

fn dense(metric: SKNeighborsMetric, rows: usize, columns: usize) -> SKSearchStrategy {
    resolve(
        SKNeighborsAlgorithm::Auto,
        metric,
        SKFeatureRepresentation::Dense,
        rows,
        columns,
    )
    .expect("the dense auto policy always resolves")
}

#[test]
fn algorithm_auto_uses_kd_tree_on_large_dense_tree_metrics() {
    for metric in [
        SKNeighborsMetric::Euclidean,
        SKNeighborsMetric::Manhattan,
        SKNeighborsMetric::Chebyshev,
    ] {
        assert_eq!(
            dense(metric, SK_AUTO_TREE_MINIMUM_TRAINING_ROWS, 3),
            SKSearchStrategy::KdTree
        );
    }
}

#[test]
fn algorithm_auto_uses_a_ball_tree_for_wide_dense_data() {
    assert_eq!(
        dense(
            SKNeighborsMetric::Euclidean,
            SK_AUTO_TREE_MINIMUM_TRAINING_ROWS,
            SK_AUTO_BALL_TREE_MINIMUM_FEATURES + 1,
        ),
        SKSearchStrategy::BallTree
    );
}

#[test]
fn algorithm_auto_falls_back_to_brute_force_below_the_row_threshold() {
    assert_eq!(
        dense(
            SKNeighborsMetric::Euclidean,
            SK_AUTO_TREE_MINIMUM_TRAINING_ROWS - 1,
            3,
        ),
        SKSearchStrategy::BruteForce
    );
}

#[test]
fn algorithm_auto_falls_back_to_brute_force_for_cosine() {
    assert_eq!(
        dense(SKNeighborsMetric::Cosine, 10_000, 3),
        SKSearchStrategy::BruteForce
    );
}

#[test]
fn algorithm_auto_falls_back_to_brute_force_for_sparse_input() {
    let strategy = resolve(
        SKNeighborsAlgorithm::Auto,
        SKNeighborsMetric::Euclidean,
        SKFeatureRepresentation::Sparse,
        10_000,
        3,
    )
    .expect("sparse auto resolves");
    assert_eq!(strategy, SKSearchStrategy::BruteForce);
    assert!(!strategy.is_tree());
}

#[test]
fn algorithm_explicit_brute_force_wins_on_dense_data() {
    assert_eq!(
        resolve(
            SKNeighborsAlgorithm::BruteForce,
            SKNeighborsMetric::Euclidean,
            SKFeatureRepresentation::Dense,
            10_000,
            3,
        )
        .unwrap(),
        SKSearchStrategy::BruteForce
    );
}

#[test]
fn algorithm_explicit_tree_requests_are_honoured_on_dense_data() {
    for (algorithm, expected) in [
        (SKNeighborsAlgorithm::KdTree, SKSearchStrategy::KdTree),
        (SKNeighborsAlgorithm::BallTree, SKSearchStrategy::BallTree),
    ] {
        assert_eq!(
            resolve(
                algorithm,
                SKNeighborsMetric::Manhattan,
                SKFeatureRepresentation::Dense,
                5,
                2,
            )
            .unwrap(),
            expected
        );
    }
}

#[test]
fn algorithm_explicit_tree_on_sparse_data_is_rejected() {
    for algorithm in [SKNeighborsAlgorithm::KdTree, SKNeighborsAlgorithm::BallTree] {
        match resolve(
            algorithm,
            SKNeighborsMetric::Euclidean,
            SKFeatureRepresentation::Sparse,
            100,
            3,
        ) {
            Err(SKError::UnsupportedRepresentation {
                representation,
                suggestion,
            }) => {
                assert_eq!(representation, "csr");
                assert!(suggestion.contains("BruteForce"));
            }
            other => panic!("expected UnsupportedRepresentation, found {other:?}"),
        }
    }
}

#[test]
fn algorithm_explicit_tree_with_cosine_is_rejected() {
    match resolve(
        SKNeighborsAlgorithm::KdTree,
        SKNeighborsMetric::Cosine,
        SKFeatureRepresentation::Dense,
        100,
        3,
    ) {
        Err(SKError::UnsupportedRepresentation {
            representation,
            suggestion,
        }) => {
            assert_eq!(representation, "cosine distance");
            assert!(suggestion.contains("BruteForce"));
        }
        other => panic!("expected UnsupportedRepresentation, found {other:?}"),
    }
}
