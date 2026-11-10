//! Companion tests for the squared-Euclidean assignment kernels.

use ndarray::{Array2, array};
use sciencekit_common::{SKDataView, SKError};

use super::sk_squared_distances_to_centers;

fn dense_fixture() -> Array2<f64> {
    array![[0.0_f64, 0.0], [1.0, 0.0], [0.0, 2.0]]
}

fn center_fixture() -> Array2<f64> {
    array![[0.0_f64, 0.0], [1.0, 1.0]]
}

#[test]
fn kmeans_distance_dense_matches_hand_computed_squared_distances() {
    let distances = sk_squared_distances_to_centers(
        SKDataView::Dense(dense_fixture().view()),
        &center_fixture().view(),
    )
    .unwrap();

    let expected = array![[0.0_f64, 2.0], [1.0, 1.0], [4.0, 2.0]];
    assert_eq!(distances, expected);
}

#[test]
fn kmeans_distance_rejects_feature_count_mismatch() {
    let centers = array![[0.0_f64, 0.0, 0.0]];
    let error = sk_squared_distances_to_centers(
        SKDataView::Dense(dense_fixture().view()),
        &centers.view(),
    )
    .unwrap_err();

    match error {
        SKError::ShapeMismatch { expected, found } => {
            assert_eq!(expected, vec![3, 3]);
            assert_eq!(found, vec![3, 2]);
        }
        other => panic!("expected ShapeMismatch, found {other:?}"),
    }
}
