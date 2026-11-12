//! Companion validation tests for `SKDBSCAN` construction, fit and prediction
//! (spec `dbscan`).

use ndarray::{Array2, array};
use sciencekit_common::{SKBuilder, SKClustererPredictor, SKError, SKUnsupervisedFit};
use sprs::{CsMat, TriMat};

use super::{SKDBSCAN, SKDBSCANAlgorithm, SKDBSCANMetric};

fn small_estimator() -> SKDBSCAN<f64> {
    SKDBSCAN::<f64>::build()
        .eps(0.6)
        .minimum_samples(2)
        .build()
        .unwrap()
}

fn csr(entries: &[(usize, usize, f64)], shape: (usize, usize)) -> CsMat<f64> {
    let mut triplet = TriMat::new(shape);
    for &(row, column, value) in entries {
        triplet.add_triplet(row, column, value);
    }
    triplet.to_csr()
}

fn assert_empty(error: SKError, expected_operation: &'static str) {
    match error {
        SKError::EmptyInput { operation } => assert_eq!(operation, expected_operation),
        other => panic!("expected EmptyInput, found {other:?}"),
    }
}

fn assert_non_finite(error: SKError, expected_position: usize) {
    match error {
        SKError::NonFiniteValue { position } => assert_eq!(position, expected_position),
        other => panic!("expected NonFiniteValue, found {other:?}"),
    }
}

fn assert_invalid_hyperparameter(error: SKError, expected_name: &'static str) {
    match error {
        SKError::InvalidHyperparameter { name, .. } => assert_eq!(name, expected_name),
        other => panic!("expected InvalidHyperparameter, found {other:?}"),
    }
}

fn assert_shape_mismatch(
    error: SKError,
    expected_dimensions: &[usize],
    found_dimensions: &[usize],
) {
    match error {
        SKError::ShapeMismatch { expected, found } => {
            assert_eq!(expected, expected_dimensions);
            assert_eq!(found, found_dimensions);
        }
        other => panic!("expected ShapeMismatch, found {other:?}"),
    }
}

fn assert_unsupported(error: SKError, expected_representation: &'static str, needle: &'static str) {
    match error {
        SKError::UnsupportedRepresentation {
            representation,
            suggestion,
        } => {
            assert_eq!(representation, expected_representation);
            assert!(suggestion.contains(needle));
        }
        other => panic!("expected UnsupportedRepresentation, found {other:?}"),
    }
}

#[test]
fn dbscan_validation_rejects_empty_training_and_prediction_inputs() {
    let estimator = small_estimator();
    for empty in [Array2::<f64>::zeros((0, 2)), Array2::<f64>::zeros((3, 0))] {
        assert_empty(estimator.fit(&empty).unwrap_err(), "fit");
        assert_empty(estimator.fit_predict(&empty).unwrap_err(), "fit");
    }
    for empty_sparse in [csr(&[], (0, 2)), csr(&[], (3, 0))] {
        assert_empty(estimator.fit(empty_sparse.view()).unwrap_err(), "fit");
    }

    let model = estimator
        .fit(&array![[0.0_f64, 0.0], [0.1, 0.0], [5.0, 5.0]])
        .unwrap();
    let empty_dense_queries = Array2::<f64>::zeros((0, 2));
    assert_empty(
        model.predict_clusters(&empty_dense_queries).unwrap_err(),
        "predict",
    );
    let empty_queries = csr(&[], (0, 2));
    assert_empty(
        model.predict_clusters(empty_queries.view()).unwrap_err(),
        "predict",
    );
}

#[test]
fn dbscan_validation_rejects_non_positive_eps_zero_minimum_samples_and_zero_leaf_size() {
    for invalid_eps in [0.0_f64, -0.25, f64::NAN, f64::INFINITY] {
        assert_invalid_hyperparameter(
            SKDBSCAN::<f64>::build()
                .eps(invalid_eps)
                .build()
                .unwrap_err(),
            "eps",
        );
    }
    assert_invalid_hyperparameter(
        SKDBSCAN::<f64>::build()
            .minimum_samples(0)
            .build()
            .unwrap_err(),
        "minimum_samples",
    );
    assert_invalid_hyperparameter(
        SKDBSCAN::<f64>::build().leaf_size(0).build().unwrap_err(),
        "leaf_size",
    );
}

#[test]
fn dbscan_validation_rejects_feature_count_mismatch_on_dense_and_sparse_queries() {
    let model = small_estimator()
        .fit(&array![[0.0_f64, 0.0], [0.1, 0.0], [5.0, 5.0]])
        .unwrap();
    let wrong_dense = Array2::<f64>::zeros((4, 3));
    assert_shape_mismatch(
        model.predict_clusters(&wrong_dense).unwrap_err(),
        &[4, 2],
        &[4, 3],
    );
    let sparse_wrong = csr(&[(0, 0, 1.0)], (4, 1));
    assert_shape_mismatch(
        model.predict_clusters(sparse_wrong.view()).unwrap_err(),
        &[4, 2],
        &[4, 1],
    );
}

#[test]
fn dbscan_validation_rejects_non_finite_values_in_training_and_queries() {
    let estimator = small_estimator();
    let not_a_number_data = array![[0.0_f64, 1.0], [2.0, f64::NAN]];
    let infinite_data = array![[0.0_f64, 1.0], [f64::INFINITY, 2.0]];
    assert_non_finite(estimator.fit(&not_a_number_data).unwrap_err(), 3);
    assert_non_finite(estimator.fit(&infinite_data).unwrap_err(), 2);

    let sparse_not_a_number = csr(&[(0, 0, 1.0), (1, 2, f64::NAN)], (3, 3));
    assert_non_finite(estimator.fit(sparse_not_a_number.view()).unwrap_err(), 5);

    let base = array![[0.0_f64, 0.0, 0.0], [0.1, 0.0, 0.0]];
    let model = estimator.fit(&base).unwrap();
    let infinite_query = array![[0.0_f64, f64::NEG_INFINITY, 0.0]];
    assert_non_finite(model.predict_clusters(&infinite_query).unwrap_err(), 1);
    assert_non_finite(
        model
            .predict_clusters(sparse_not_a_number.view())
            .unwrap_err(),
        5,
    );
}

#[test]
fn dbscan_validation_rejects_unsupported_metric_algorithm_and_representation_pairs() {
    let sparse = csr(&[(0, 0, 1.0), (1, 1, 2.0), (2, 0, 1.1)], (3, 2));
    for tree in [SKDBSCANAlgorithm::KdTree, SKDBSCANAlgorithm::BallTree] {
        let cosine_error = SKDBSCAN::<f64>::build()
            .algorithm(tree)
            .metric(SKDBSCANMetric::Cosine)
            .build()
            .unwrap_err();
        assert_unsupported(cosine_error, "cosine distance", "BruteForce");
        let tree_estimator = SKDBSCAN::<f64>::build().algorithm(tree).build().unwrap();
        assert_unsupported(
            tree_estimator.fit(sparse.view()).unwrap_err(),
            "csr",
            "BruteForce",
        );
    }

    let csc_matrix = csr(&[(0, 0, 1.0), (1, 1, 2.0)], (2, 3)).to_csc();
    let csc_fit_error = small_estimator().fit(csc_matrix.view()).unwrap_err();
    assert_unsupported(csc_fit_error, "csc", "CSR");
    let base = array![[0.0_f64, 0.0, 0.0], [0.1, 0.0, 0.0]];
    let model = small_estimator().fit(&base).unwrap();
    assert_unsupported(
        model.predict_clusters(csc_matrix.view()).unwrap_err(),
        "csc",
        "CSR",
    );
}
