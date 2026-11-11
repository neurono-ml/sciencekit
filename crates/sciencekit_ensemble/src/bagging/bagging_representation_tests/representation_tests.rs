//! Dense/CSR pass-through and dense-only rejection tests.

use std::sync::{Arc, Mutex};

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKError, SKRegressorPredictor, SKSupervisedFit,
};
use sciencekit_linear_model::linear_regression::core_implementation::SKLinearRegression;
use sciencekit_neighbors::{SKKNeighborsClassifier, SKNeighborsAlgorithm};
use sciencekit_tree::{SKDecisionTreeClassifier, SKDecisionTreeRegressor, SKMaximumFeatures};
use sprs::TriMat;

use crate::{SKBaggingClassifier, SKBaggingError, SKBaggingRegressor};

use super::fixtures::{
    CustomDenseOnlyClassifier, CustomDenseOnlyError, RepresentationSpyRegressor,
};

#[test]
fn sparse_capable_base_receives_csr_without_densifying_and_matches_dense() {
    let dense = Array2::from_shape_fn((24, 4), |(row, column)| {
        if (row + column) % 2 == 0 {
            0.0
        } else if row < 12 {
            -2.0 - (column as f64)
        } else {
            2.0 + (column as f64)
        }
    });
    let mut triplets = TriMat::new((24, 4));
    dense
        .indexed_iter()
        .filter(|entry| *entry.1 != 0.0)
        .for_each(|((row, column), &value)| triplets.add_triplet(row, column, value));
    let sparse = triplets.to_csr::<usize>();
    let labels = Array1::from_shape_fn(24, |row| i64::from(row >= 12));
    let targets = Array1::from_shape_fn(24, |row| if row < 12 { -3.0 } else { 3.0 });

    let fit_representations = Arc::new(Mutex::new(Vec::new()));
    let predict_representations = Arc::new(Mutex::new(Vec::new()));
    let spy = RepresentationSpyRegressor {
        seen_fit_representations: Arc::clone(&fit_representations),
        seen_predict_representations: Arc::clone(&predict_representations),
    };
    let spy_model = SKBaggingRegressor::build(spy)
        .number_of_estimators(4)
        .maximum_features(SKMaximumFeatures::Count(3))
        .oob_score(true)
        .random_seed(5)
        .build()
        .unwrap()
        .fit(sparse.view(), targets.view())
        .unwrap();
    let _ = spy_model.predict(sparse.view()).unwrap();
    assert!(
        fit_representations
            .lock()
            .unwrap()
            .iter()
            .all(|&representation| representation == "csr")
    );
    assert!(
        predict_representations
            .lock()
            .unwrap()
            .iter()
            .all(|&representation| representation == "csr")
    );

    let base_tree = SKDecisionTreeClassifier::<f64>::build().build().unwrap();
    let dense_model = SKBaggingClassifier::build(base_tree.clone())
        .number_of_estimators(8)
        .maximum_features(SKMaximumFeatures::Count(3))
        .oob_score(true)
        .random_seed(13)
        .build()
        .unwrap()
        .fit(dense.view(), labels.view())
        .unwrap();
    let sparse_model = SKBaggingClassifier::build(base_tree)
        .number_of_estimators(8)
        .maximum_features(SKMaximumFeatures::Count(3))
        .oob_score(true)
        .random_seed(13)
        .build()
        .unwrap()
        .fit(sparse.view(), labels.view())
        .unwrap();

    assert_eq!(
        dense_model.predict_labels(dense.view()).unwrap(),
        sparse_model.predict_labels(sparse.view()).unwrap()
    );
    assert_eq!(
        dense_model.predict_labels(sparse.view()).unwrap(),
        sparse_model.predict_labels(dense.view()).unwrap()
    );
    assert!((dense_model.oob_score().unwrap() - sparse_model.oob_score().unwrap()).abs() < 1e-12);

    // Ultra-wide 64 x 100_000 CSR matrix (2 non-zeros per row) never densifies.
    let mut wide_triplets = TriMat::new((64, 100_000));
    (0..64).for_each(|row| {
        let sign = if row < 32 { -1.0 } else { 1.0 };
        wide_triplets.add_triplet(row, 7, sign * 3.0);
        wide_triplets.add_triplet(row, 99_991, sign * 5.0);
    });
    let wide_sparse = wide_triplets.to_csr::<usize>();
    let wide_targets = Array1::from_shape_fn(64, |row| if row < 32 { -4.0 } else { 4.0 });
    let regression_tree = SKDecisionTreeRegressor::<f64>::build()
        .maximum_depth(3)
        .build()
        .unwrap();
    let wide_predictions = SKBaggingRegressor::build(regression_tree)
        .number_of_estimators(4)
        .random_seed(9)
        .build()
        .unwrap()
        .fit(wide_sparse.view(), wide_targets.view())
        .unwrap()
        .predict(wide_sparse.view())
        .unwrap();
    assert_eq!(wide_predictions.len(), 64);
}

#[test]
fn dense_only_base_estimators_surface_error_through_sk_bagging_error_base() {
    let mut triplets = TriMat::new((8, 2));
    (0..8).for_each(|row| triplets.add_triplet(row, 0, (row + 1) as f64));
    let sparse = triplets.to_csr::<usize>();
    let dense = sparse.to_dense();
    let labels = Array1::from_vec(vec![0_i64, 0, 0, 0, 1, 1, 1, 1]);
    let targets = Array1::from_shape_fn(8, |row| (row as f64) * 2.0);

    let custom_fit_error = SKBaggingClassifier::build(CustomDenseOnlyClassifier)
        .number_of_estimators(3)
        .build()
        .unwrap()
        .fit(sparse.view(), labels.view())
        .unwrap_err();
    assert!(custom_fit_error.is_base());
    assert_eq!(
        custom_fit_error.as_base(),
        Some(&CustomDenseOnlyError::RejectedRepresentation("csr"))
    );

    let fitted_on_dense = SKBaggingClassifier::build(CustomDenseOnlyClassifier)
        .number_of_estimators(3)
        .build()
        .unwrap()
        .fit(dense.view(), labels.view())
        .unwrap();
    let custom_predict_error = fitted_on_dense
        .predict_probabilities(sparse.view())
        .unwrap_err();
    assert!(custom_predict_error.is_base());
    assert_eq!(
        custom_predict_error.as_base(),
        Some(&CustomDenseOnlyError::RejectedRepresentation("csr"))
    );

    let linear_error = SKBaggingRegressor::build(SKLinearRegression::build().build().unwrap())
        .number_of_estimators(2)
        .build()
        .unwrap()
        .fit(sparse.view(), targets.view())
        .unwrap_err();
    assert!(matches!(
        linear_error,
        SKBaggingError::Base(SKError::UnsupportedRepresentation {
            representation: "csr",
            ..
        })
    ));

    let kd_neighbors = SKKNeighborsClassifier::<f64>::build()
        .algorithm(SKNeighborsAlgorithm::KdTree)
        .build()
        .unwrap();
    let kd_error = SKBaggingClassifier::build(kd_neighbors)
        .number_of_estimators(2)
        .build()
        .unwrap()
        .fit(sparse.view(), labels.view())
        .unwrap_err();
    assert!(matches!(
        kd_error,
        SKBaggingError::Base(SKError::UnsupportedRepresentation {
            representation: "csr",
            ..
        })
    ));
}
