//! Companion tests for regressor representation parity
//! (spec `nearest-neighbor-estimators`).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKRegressorPredictor, SKSupervisedFit};
use sprs::{CsMat, TriMat};

use super::SKKNeighborsRegressor;
use crate::neighbors::SKSearchStrategy;

const ROWS: usize = 80;

fn dense_design() -> Array2<f64> {
    Array2::from_shape_fn((ROWS, 4), |(row, column)| {
        if (row + column) % 3 == 0 {
            0.0
        } else {
            ((row * 7 + column * 13) % 19) as f64 / 4.0
        }
    })
}

fn sparse_design(dense: &Array2<f64>) -> CsMat<f64> {
    let mut triplet = TriMat::new((dense.nrows(), dense.ncols()));
    for ((row, column), value) in dense.indexed_iter() {
        if *value != 0.0 {
            triplet.add_triplet(row, column, *value);
        }
    }
    triplet.to_csr()
}

fn responses() -> Array1<f64> {
    Array1::from_shape_fn(ROWS, |row| (row % 5) as f64 * 0.5 + 1.0)
}

#[test]
fn representation_sparse_and_dense_regressor_predictions_agree() {
    let dense = dense_design();
    let responses = responses();
    let dense_model = SKKNeighborsRegressor::<f64>::build()
        .build()
        .unwrap()
        .fit(dense.view(), responses.view())
        .unwrap();
    let sparse_model = SKKNeighborsRegressor::<f64>::build()
        .build()
        .unwrap()
        .fit(sparse_design(&dense).view(), responses.view())
        .unwrap();

    let dense_predictions = dense_model.predict(dense.view()).unwrap();
    let sparse_predictions = sparse_model.predict(dense.view()).unwrap();

    for (dense, sparse) in dense_predictions.iter().zip(sparse_predictions.iter()) {
        assert!((dense - sparse).abs() < 1e-12);
    }
    assert_eq!(dense_model.search_strategy(), SKSearchStrategy::KdTree);
    assert_eq!(sparse_model.search_strategy(), SKSearchStrategy::BruteForce);
    assert_eq!(sparse_model.training_representation(), "csr");
    assert_eq!(dense_model.training_representation(), "dense");
}

#[test]
fn representation_distance_weighted_sparse_and_dense_agree() {
    let dense = dense_design();
    let responses = responses();
    let dense_model = SKKNeighborsRegressor::<f64>::build()
        .weights(crate::neighbors::SKNeighborsWeights::Distance)
        .build()
        .unwrap()
        .fit(dense.view(), responses.view())
        .unwrap();
    let sparse_model = SKKNeighborsRegressor::<f64>::build()
        .weights(crate::neighbors::SKNeighborsWeights::Distance)
        .build()
        .unwrap()
        .fit(sparse_design(&dense).view(), responses.view())
        .unwrap();

    let dense_predictions = dense_model.predict(dense.view()).unwrap();
    let sparse_predictions = sparse_model.predict(dense.view()).unwrap();

    for (dense, sparse) in dense_predictions.iter().zip(sparse_predictions.iter()) {
        assert!((dense - sparse).abs() < 1e-12);
    }
}

#[test]
fn representation_dense_regressor_predicts_on_a_sparse_query() {
    let dense = dense_design();
    let responses = responses();
    let model = SKKNeighborsRegressor::<f64>::build()
        .build()
        .unwrap()
        .fit(dense.view(), responses.view())
        .unwrap();
    let sparse_queries = sparse_design(&dense);

    let dense_predictions = model.predict(dense.view()).unwrap();
    let sparse_predictions = model.predict(sparse_queries.view()).unwrap();

    for (dense, sparse) in dense_predictions.iter().zip(sparse_predictions.iter()) {
        assert!((dense - sparse).abs() < 1e-12);
    }
    assert_eq!(model.search_strategy(), SKSearchStrategy::KdTree);
}

#[test]
fn representation_sparse_regressor_predicts_on_a_dense_query() {
    let dense = dense_design();
    let responses = responses();
    let sparse_model = SKKNeighborsRegressor::<f64>::build()
        .build()
        .unwrap()
        .fit(sparse_design(&dense).view(), responses.view())
        .unwrap();
    let dense_model = SKKNeighborsRegressor::<f64>::build()
        .build()
        .unwrap()
        .fit(dense.view(), responses.view())
        .unwrap();

    let from_sparse_model = sparse_model.predict(dense.view()).unwrap();
    let from_dense_model = dense_model.predict(dense.view()).unwrap();

    for (left, right) in from_sparse_model.iter().zip(from_dense_model.iter()) {
        assert!((left - right).abs() < 1e-12);
    }
    assert_eq!(sparse_model.training_representation(), "csr");
}
