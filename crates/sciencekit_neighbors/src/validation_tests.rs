//! Companion tests for metric, finiteness and shape validation.

use ndarray::array;
use sciencekit_common::{SKDataView, SKError};

use crate::ball_tree::SKBallTreeBuilder;
use crate::kd_tree::SKKDTreeBuilder;
use crate::metric::SKRequestedNeighborMetric;
use crate::searcher::SKNearestNeighborsSearcher;

fn finite_data() -> ndarray::Array2<f64> {
    array![[0.0_f64, 1.0], [2.0, 3.0], [4.0, 5.0]]
}

fn assert_invalid_metric(error: SKError) {
    match error {
        SKError::InvalidHyperparameter { name, reason } => {
            assert_eq!(name, "metric");
            assert!(!reason.is_empty());
        }
        other => panic!("expected InvalidHyperparameter, found {other:?}"),
    }
}

#[test]
fn validation_rejects_cosine_and_nan_euclidean_on_both_builders() {
    let data = finite_data();
    assert_invalid_metric(
        SKKDTreeBuilder::<f64>::new()
            .metric(SKRequestedNeighborMetric::Cosine)
            .build(SKDataView::Dense(data.view()))
            .unwrap_err(),
    );
    assert_invalid_metric(
        SKKDTreeBuilder::<f64>::new()
            .metric(SKRequestedNeighborMetric::NanEuclidean)
            .build(SKDataView::Dense(data.view()))
            .unwrap_err(),
    );
    assert_invalid_metric(
        SKBallTreeBuilder::<f64>::new()
            .metric(SKRequestedNeighborMetric::Cosine)
            .build(SKDataView::Dense(data.view()))
            .unwrap_err(),
    );
    assert_invalid_metric(
        SKBallTreeBuilder::<f64>::new()
            .metric(SKRequestedNeighborMetric::NanEuclidean)
            .build(SKDataView::Dense(data.view()))
            .unwrap_err(),
    );
}

#[test]
fn validation_rejects_non_finite_training_values_at_their_flat_position() {
    let not_a_number = array![[0.0_f64, 1.0], [2.0, f64::NAN]];
    let infinite = array![[0.0_f64, 1.0], [f64::INFINITY, 2.0]];

    for (data, position) in [(not_a_number, 3), (infinite, 2)] {
        for error in [
            SKKDTreeBuilder::<f64>::new()
                .build(SKDataView::Dense(data.view()))
                .unwrap_err(),
            SKBallTreeBuilder::<f64>::new()
                .build(SKDataView::Dense(data.view()))
                .unwrap_err(),
        ] {
            match error {
                SKError::NonFiniteValue { position: found } => assert_eq!(found, position),
                other => panic!("expected NonFiniteValue, found {other:?}"),
            }
        }
    }
}

#[test]
fn validation_rejects_non_finite_query_values_at_their_flat_position() {
    let data = finite_data();
    let kd = SKKDTreeBuilder::<f64>::new()
        .build(SKDataView::Dense(data.view()))
        .unwrap();
    let queries = array![[0.0_f64, 1.0], [2.0, f64::NEG_INFINITY]];

    match kd.k_nearest(SKDataView::Dense(queries.view()), 1).unwrap_err() {
        SKError::NonFiniteValue { position } => assert_eq!(position, 3),
        other => panic!("expected NonFiniteValue, found {other:?}"),
    }
}

#[test]
fn validation_rejects_a_query_with_the_wrong_feature_count() {
    let data = finite_data();
    let kd = SKKDTreeBuilder::<f64>::new()
        .build(SKDataView::Dense(data.view()))
        .unwrap();
    let queries = array![[0.0_f64, 1.0, 2.0]];

    match kd.k_nearest(SKDataView::Dense(queries.view()), 1).unwrap_err() {
        SKError::ShapeMismatch { expected, found } => {
            assert_eq!(expected, vec![1, 2]);
            assert_eq!(found, vec![1, 3]);
        }
        other => panic!("expected ShapeMismatch, found {other:?}"),
    }
}
