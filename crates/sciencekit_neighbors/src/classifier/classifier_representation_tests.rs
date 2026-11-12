//! Companion tests for classifier representation parity
//! (spec `nearest-neighbor-estimators`).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKError, SKSupervisedFit};
use sprs::CsMat;

use super::SKKNeighborsClassifier;
use crate::neighbors::{SKNeighborsAlgorithm, SKNeighborsMetric, SKSearchStrategy};

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
    use sprs::TriMat;
    let mut triplet = TriMat::new((dense.nrows(), dense.ncols()));
    for ((row, column), value) in dense.indexed_iter() {
        if *value != 0.0 {
            triplet.add_triplet(row, column, *value);
        }
    }
    triplet.to_csr()
}

fn fitted_sparse() -> (Array2<f64>, super::SKKNeighborsClassifierModel<f64>) {
    let dense = dense_design();
    let labels = Array1::from_shape_fn(ROWS, |row| (row % 3) as i64);
    let model = SKKNeighborsClassifier::<f64>::build()
        .build()
        .unwrap()
        .fit(sparse_design(&dense).view(), labels.view())
        .unwrap();
    (dense, model)
}

#[test]
fn representation_sparse_and_dense_classifier_predictions_agree() {
    let dense = dense_design();
    let labels = Array1::from_shape_fn(ROWS, |row| (row % 3) as i64);
    let dense_model = SKKNeighborsClassifier::<f64>::build()
        .build()
        .unwrap()
        .fit(dense.view(), labels.view())
        .unwrap();
    let sparse_model = SKKNeighborsClassifier::<f64>::build()
        .build()
        .unwrap()
        .fit(sparse_design(&dense).view(), labels.view())
        .unwrap();

    let dense_labels = dense_model.predict_labels(dense.view()).unwrap();
    let sparse_labels = sparse_model.predict_labels(dense.view()).unwrap();

    assert_eq!(dense_labels, sparse_labels);
    assert_eq!(dense_model.search_strategy(), SKSearchStrategy::KdTree);
    assert_eq!(sparse_model.search_strategy(), SKSearchStrategy::BruteForce);
    assert_eq!(sparse_model.training_representation(), "csr");
    assert_eq!(dense_model.training_representation(), "dense");
}

#[test]
fn representation_sparse_classifier_probabilities_agree_within_tolerance() {
    let dense = dense_design();
    let labels = Array1::from_shape_fn(ROWS, |row| (row % 3) as i64);
    let dense_model = SKKNeighborsClassifier::<f64>::build()
        .build()
        .unwrap()
        .fit(dense.view(), labels.view())
        .unwrap();
    let (_, sparse_model) = fitted_sparse();

    let dense_probabilities = dense_model.predict_probabilities(dense.view()).unwrap();
    let sparse_probabilities = sparse_model.predict_probabilities(dense.view()).unwrap();

    assert_eq!(dense_probabilities.dim(), sparse_probabilities.dim());
    for (dense, sparse) in dense_probabilities.iter().zip(sparse_probabilities.iter()) {
        assert!((dense - sparse).abs() < 1e-12);
    }
}

#[test]
fn representation_dense_model_predicts_on_a_sparse_query() {
    let dense = dense_design();
    let labels = Array1::from_shape_fn(ROWS, |row| (row % 3) as i64);
    let model = SKKNeighborsClassifier::<f64>::build()
        .build()
        .unwrap()
        .fit(dense.view(), labels.view())
        .unwrap();
    let sparse_queries = sparse_design(&dense);

    let dense_labels = model.predict_labels(dense.view()).unwrap();
    let sparse_labels = model.predict_labels(sparse_queries.view()).unwrap();

    assert_eq!(dense_labels, sparse_labels);
    assert_eq!(model.search_strategy(), SKSearchStrategy::KdTree);
}

#[test]
fn representation_sparse_model_predicts_on_a_dense_query() {
    let dense = dense_design();
    let labels = Array1::from_shape_fn(ROWS, |row| (row % 3) as i64);
    let sparse_model = SKKNeighborsClassifier::<f64>::build()
        .build()
        .unwrap()
        .fit(sparse_design(&dense).view(), labels.view())
        .unwrap();
    let dense_model = SKKNeighborsClassifier::<f64>::build()
        .build()
        .unwrap()
        .fit(dense.view(), labels.view())
        .unwrap();

    let from_sparse_model = sparse_model.predict_labels(dense.view()).unwrap();
    let from_dense_model = dense_model.predict_labels(dense.view()).unwrap();

    assert_eq!(from_sparse_model, from_dense_model);
    assert_eq!(sparse_model.training_representation(), "csr");
}

#[test]
fn representation_auto_on_sparse_reports_brute_force_without_densifying() {
    let (_, model) = fitted_sparse();

    assert_eq!(model.search_strategy(), SKSearchStrategy::BruteForce);
    assert_eq!(model.training_representation(), "csr");
    assert_eq!(model.training_rows(), ROWS);
}

#[test]
fn representation_explicit_trees_over_sparse_are_rejected() {
    let dense = dense_design();
    let labels = Array1::from_shape_fn(ROWS, |row| (row % 3) as i64);
    for algorithm in [SKNeighborsAlgorithm::KdTree, SKNeighborsAlgorithm::BallTree] {
        let outcome = SKKNeighborsClassifier::<f64>::build()
            .algorithm(algorithm)
            .build()
            .unwrap()
            .fit(sparse_design(&dense).view(), labels.view());

        match outcome {
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
fn representation_auto_on_cosine_reports_brute_force() {
    let dense = dense_design();
    let labels = Array1::from_shape_fn(ROWS, |row| (row % 3) as i64);
    let model = SKKNeighborsClassifier::<f64>::build()
        .metric(SKNeighborsMetric::Cosine)
        .build()
        .unwrap()
        .fit(dense.view(), labels.view())
        .unwrap();

    assert_eq!(model.search_strategy(), SKSearchStrategy::BruteForce);
    let predictions = model.predict_labels(dense.view()).unwrap();
    assert_eq!(predictions.len(), ROWS);
}
