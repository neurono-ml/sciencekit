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

#[test]
fn dbscan_validation_rejects_empty_training_and_prediction_inputs() {
    let estimator = small_estimator();
    for empty in [Array2::<f64>::zeros((0, 2)), Array2::<f64>::zeros((3, 0))] {
        match estimator.fit(&empty).unwrap_err() {
            SKError::EmptyInput { operation } => assert_eq!(operation, "fit"),
            other => panic!("expected EmptyInput, found {other:?}"),
        }
        match estimator.fit_predict(&empty).unwrap_err() {
            SKError::EmptyInput { operation } => assert_eq!(operation, "fit"),
            other => panic!("expected EmptyInput, found {other:?}"),
        }
    }
    for empty_sparse in [csr(&[], (0, 2)), csr(&[], (3, 0))] {
        match estimator.fit(empty_sparse.view()).unwrap_err() {
            SKError::EmptyInput { operation } => assert_eq!(operation, "fit"),
            other => panic!("expected EmptyInput, found {other:?}"),
        }
    }

    let model = estimator
        .fit(&array![[0.0_f64, 0.0], [0.1, 0.0], [5.0, 5.0]])
        .unwrap();
    match model
        .predict_clusters(&Array2::<f64>::zeros((0, 2)))
        .unwrap_err()
    {
        SKError::EmptyInput { operation } => assert_eq!(operation, "predict"),
        other => panic!("expected EmptyInput, found {other:?}"),
    }
    let empty_queries = csr(&[], (0, 2));
    match model.predict_clusters(empty_queries.view()).unwrap_err() {
        SKError::EmptyInput { operation } => assert_eq!(operation, "predict"),
        other => panic!("expected EmptyInput, found {other:?}"),
    }
}

#[test]
fn dbscan_validation_rejects_non_positive_eps_zero_minimum_samples_and_zero_leaf_size() {
    for invalid_eps in [0.0_f64, -0.25, f64::NAN, f64::INFINITY] {
        match SKDBSCAN::<f64>::build()
            .eps(invalid_eps)
            .build()
            .unwrap_err()
        {
            SKError::InvalidHyperparameter { name, .. } => assert_eq!(name, "eps"),
            other => panic!("expected InvalidHyperparameter, found {other:?}"),
        }
    }
    match SKDBSCAN::<f64>::build()
        .minimum_samples(0)
        .build()
        .unwrap_err()
    {
        SKError::InvalidHyperparameter { name, .. } => assert_eq!(name, "minimum_samples"),
        other => panic!("expected InvalidHyperparameter, found {other:?}"),
    }
    match SKDBSCAN::<f64>::build().leaf_size(0).build().unwrap_err() {
        SKError::InvalidHyperparameter { name, .. } => assert_eq!(name, "leaf_size"),
        other => panic!("expected InvalidHyperparameter, found {other:?}"),
    }
}

#[test]
fn dbscan_validation_rejects_feature_count_mismatch_on_dense_and_sparse_queries() {
    let model = small_estimator()
        .fit(&array![[0.0_f64, 0.0], [0.1, 0.0], [5.0, 5.0]])
        .unwrap();

    let dense_wrong = Array2::<f64>::zeros((4, 3));
    match model.predict_clusters(&dense_wrong).unwrap_err() {
        SKError::ShapeMismatch { expected, found } => {
            assert_eq!(expected, vec![4, 2]);
            assert_eq!(found, vec![4, 3]);
        }
        other => panic!("expected ShapeMismatch, found {other:?}"),
    }

    let sparse_wrong = csr(&[(0, 0, 1.0)], (4, 1));
    match model.predict_clusters(sparse_wrong.view()).unwrap_err() {
        SKError::ShapeMismatch { expected, found } => {
            assert_eq!(expected, vec![4, 2]);
            assert_eq!(found, vec![4, 1]);
        }
        other => panic!("expected ShapeMismatch, found {other:?}"),
    }
}

#[test]
fn dbscan_validation_rejects_non_finite_values_in_training_and_queries() {
    let estimator = small_estimator();
    let dense_nan = array![[0.0_f64, 1.0], [2.0, f64::NAN]];
    let dense_inf = array![[0.0_f64, 1.0], [f64::INFINITY, 2.0]];
    for (bad, expected_position) in [(dense_nan, 3_usize), (dense_inf, 2)] {
        match estimator.fit(&bad).unwrap_err() {
            SKError::NonFiniteValue { position } => assert_eq!(position, expected_position),
            other => panic!("expected NonFiniteValue, found {other:?}"),
        }
    }

    let sparse_nan = csr(&[(0, 0, 1.0), (1, 2, f64::NAN)], (3, 3));
    match estimator.fit(sparse_nan.view()).unwrap_err() {
        SKError::NonFiniteValue { position } => assert_eq!(position, 5),
        other => panic!("expected NonFiniteValue, found {other:?}"),
    }

    let model = estimator
        .fit(&array![[0.0_f64, 0.0, 0.0], [0.1, 0.0, 0.0]])
        .unwrap();
    let query_inf = array![[0.0_f64, f64::NEG_INFINITY, 0.0]];
    match model.predict_clusters(&query_inf).unwrap_err() {
        SKError::NonFiniteValue { position } => assert_eq!(position, 1),
        other => panic!("expected NonFiniteValue, found {other:?}"),
    }
    match model.predict_clusters(sparse_nan.view()).unwrap_err() {
        SKError::NonFiniteValue { position } => assert_eq!(position, 5),
        other => panic!("expected NonFiniteValue, found {other:?}"),
    }
}

#[test]
fn dbscan_validation_rejects_unsupported_metric_algorithm_and_representation_pairs() {
    for tree in [SKDBSCANAlgorithm::KdTree, SKDBSCANAlgorithm::BallTree] {
        match SKDBSCAN::<f64>::build()
            .algorithm(tree)
            .metric(SKDBSCANMetric::Cosine)
            .build()
            .unwrap_err()
        {
            SKError::UnsupportedRepresentation {
                representation,
                suggestion,
            } => {
                assert_eq!(representation, "cosine distance");
                assert!(suggestion.contains("BruteForce"));
            }
            other => panic!("expected UnsupportedRepresentation, found {other:?}"),
        }

        let sparse = csr(&[(0, 0, 1.0), (1, 1, 2.0), (2, 0, 1.1)], (3, 2));
        match SKDBSCAN::<f64>::build()
            .algorithm(tree)
            .build()
            .unwrap()
            .fit(sparse.view())
            .unwrap_err()
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
    }

    // CSC (compressed-column) views are rejected with guidance to convert to CSR.
    let csc_matrix = csr(&[(0, 0, 1.0), (1, 1, 2.0)], (2, 3)).to_csc();
    match small_estimator().fit(csc_matrix.view()).unwrap_err() {
        SKError::UnsupportedRepresentation {
            representation,
            suggestion,
        } => {
            assert_eq!(representation, "csc");
            assert!(suggestion.contains("CSR"));
        }
        other => panic!("expected UnsupportedRepresentation for CSC, found {other:?}"),
    }
    let model = small_estimator()
        .fit(&array![[0.0_f64, 0.0, 0.0], [0.1, 0.0, 0.0]])
        .unwrap();
    match model.predict_clusters(csc_matrix.view()).unwrap_err() {
        SKError::UnsupportedRepresentation {
            representation,
            suggestion,
        } => {
            assert_eq!(representation, "csc");
            assert!(suggestion.contains("CSR"));
        }
        other => panic!("expected UnsupportedRepresentation for CSC, found {other:?}"),
    }
}
