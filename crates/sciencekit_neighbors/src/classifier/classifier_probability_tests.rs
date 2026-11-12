//! Companion tests for classifier probability estimates
//! (spec `nearest-neighbor-estimators`).

use ndarray::{Array1, Array2, array};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKSupervisedFit};

use super::SKKNeighborsClassifier;
use crate::neighbors::SKNeighborsWeights;

fn fit_fixture(
    features: Array2<f64>,
    labels: Array1<i64>,
) -> super::SKKNeighborsClassifierModel<f64> {
    SKKNeighborsClassifier::<f64>::build()
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap()
}

fn assert_close(left: f64, right: f64) {
    assert!(
        (left - right).abs() < 1e-12,
        "expected {right}, found {left}"
    );
}

#[test]
fn classifier_probabilities_are_neighbor_fractions() {
    let features = array![[1.0_f64], [2.0], [3.0], [4.0], [5.0]];
    let labels = array![0_i64, 0, 0, 1, 1];
    let model = fit_fixture(features, labels);
    let queries = array![[2.6_f64]];

    let probabilities = model.predict_probabilities(queries.view()).unwrap();

    assert_eq!(probabilities.shape(), &[1, 2]);
    assert_close(probabilities[[0, 0]], 0.6);
    assert_close(probabilities[[0, 1]], 0.4);
    assert_close(probabilities.row(0).sum(), 1.0);
}

#[test]
fn classifier_probabilities_keep_absent_classes_at_zero() {
    let features = Array2::from_shape_fn(
        (10, 1),
        |(row, _)| {
            if row == 9 { 100.0 } else { row as f64 + 1.0 }
        },
    );
    let labels = Array1::from_shape_fn(10, |row| if row == 9 { 1 } else { 0 });
    let model = SKKNeighborsClassifier::<f64>::build()
        .nearest_neighbors_count(3)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let queries = array![[1.0_f64]];

    let probabilities = model.predict_probabilities(queries.view()).unwrap();

    assert_eq!(probabilities.shape(), &[1, 2]);
    assert_eq!(probabilities[[0, 1]], 0.0);
    assert_close(probabilities.row(0).sum(), 1.0);
}

#[test]
fn classifier_probabilities_sum_to_one_with_distance_weights() {
    let features = array![[1.0_f64], [2.0], [7.0], [8.0]];
    let labels = array![0_i64, 0, 1, 1];
    let model = SKKNeighborsClassifier::<f64>::build()
        .weights(SKNeighborsWeights::Distance)
        .nearest_neighbors_count(4)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let queries = array![[1.0_f64], [7.5]];

    let probabilities = model.predict_probabilities(queries.view()).unwrap();

    for row in 0..2 {
        assert_close(probabilities.row(row).sum(), 1.0);
    }
    assert!(probabilities[[0, 0]] > probabilities[[0, 1]]);
    assert!(probabilities[[1, 1]] > probabilities[[1, 0]]);
}

#[test]
fn classifier_probabilities_include_duplicate_rows() {
    let features = array![[1.0_f64], [1.0], [1.0], [5.0]];
    let labels = array![1_i64, 1, 0, 0];
    let model = SKKNeighborsClassifier::<f64>::build()
        .weights(SKNeighborsWeights::Distance)
        .nearest_neighbors_count(3)
        .build()
        .unwrap()
        .fit(features.view(), labels.view())
        .unwrap();
    let queries = array![[1.0_f64]];

    let probabilities = model.predict_probabilities(queries.view()).unwrap();

    // Zero-distance duplicates: two class-1 rows at distance zero and one
    // class-0 row at distance zero; every farther row is ignored. Label 1 is
    // canonical index 0 because it was seen first.
    assert_close(probabilities[[0, 0]], 2.0 / 3.0);
    assert_close(probabilities[[0, 1]], 1.0 / 3.0);
}
