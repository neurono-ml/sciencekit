//! Large-data acceptance for the neighbor estimators: at least 20 000 rows.

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit};

use crate::neighbors::SKNeighborsWeights;
use crate::{SKKNeighborsClassifier, SKKNeighborsRegressor};

const ROWS: usize = 20_000;
const COLUMNS: usize = 4;
const QUERIES: usize = 64;

fn design() -> Array2<f64> {
    Array2::from_shape_fn((ROWS, COLUMNS), |(row, column)| {
        ((row * 17 + column * 31) % 101) as f64 / 50.0 - 1.0
    })
}

fn queries() -> Array2<f64> {
    Array2::from_shape_fn((QUERIES, COLUMNS), |(row, column)| {
        ((row * 13 + column * 7) % 97) as f64 / 48.0 - 1.0
    })
}

fn labels() -> Array1<i64> {
    Array1::from_shape_fn(ROWS, |row| {
        let score = (row as f64).sin();
        if score > 0.0 { 1 } else { 0 }
    })
}

#[test]
fn acceptance_knn_large_dense_classifier_predicts_the_whole_batch() {
    let features = design();
    let model = SKKNeighborsClassifier::<f64>::build()
        .nearest_neighbors_count(7)
        .build()
        .unwrap()
        .fit(features.view(), labels().view())
        .unwrap();

    let predictions = model.predict_labels(queries().view()).unwrap();

    assert_eq!(predictions.len(), QUERIES);
    assert!(predictions.iter().all(|label| *label == 0 || *label == 1));
}

#[test]
fn acceptance_knn_large_distance_weighted_regressor_stays_finite() {
    let features = design();
    let responses = Array1::from_shape_fn(ROWS, |row| {
        0.5 * (row as f64).cos() + 0.25 * (row as f64).sin()
    });
    let model = SKKNeighborsRegressor::<f64>::build()
        .weights(SKNeighborsWeights::Distance)
        .nearest_neighbors_count(9)
        .build()
        .unwrap()
        .fit(features.view(), responses.view())
        .unwrap();

    let predictions = model.predict(queries().view()).unwrap();

    assert_eq!(predictions.len(), QUERIES);
    assert!(predictions.iter().all(|value| value.is_finite()));
}

#[test]
fn acceptance_knn_large_sparse_brute_force_path_completes() {
    use sprs::TriMat;

    let features = design();
    let mut triplet = TriMat::new((ROWS, COLUMNS));
    for ((row, column), value) in features.indexed_iter() {
        if *value != 0.0 {
            triplet.add_triplet(row, column, *value);
        }
    }
    let model = SKKNeighborsClassifier::<f64>::build()
        .nearest_neighbors_count(5)
        .build()
        .unwrap()
        .fit(triplet.to_csr().view(), labels().view())
        .unwrap();

    let predictions = model.predict_labels(queries().view()).unwrap();

    assert_eq!(predictions.len(), QUERIES);
    assert_eq!(model.training_representation(), "csr");
}
