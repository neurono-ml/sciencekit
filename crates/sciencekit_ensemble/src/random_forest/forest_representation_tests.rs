//! Companion tests for dense/sparse representation parity and non-densifying
//! sparse fitting (task 3.3).

use ndarray::{Array1, Array2, array};
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKError, SKRegressorPredictor, SKSupervisedFit,
};
use sprs::CsMat;

use crate::bootstrap::sk_gather_sparse_rows;

use super::{SKRandomForestClassifier, SKRandomForestRegressor};

fn logical_fixture() -> (Array2<f64>, CsMat<f64>, Array1<i64>, Array1<f64>) {
    let dense = array![
        [0.0, 2.0, 0.0, 1.0],
        [0.0, 1.5, 0.0, 0.0],
        [0.0, 2.5, 0.0, 0.5],
        [3.0, 0.0, 0.0, 0.0],
        [4.0, 0.0, 1.0, 0.0],
        [3.5, 0.0, 0.5, 0.0],
    ];
    let sparse = CsMat::new(
        (6, 4),
        vec![0, 2, 3, 5, 6, 8, 10],
        vec![1, 3, 1, 1, 3, 0, 0, 2, 0, 2],
        vec![2.0, 1.0, 1.5, 2.5, 0.5, 3.0, 4.0, 1.0, 3.5, 0.5],
    );
    let labels = array![0_i64, 0, 0, 1, 1, 1];
    let responses = array![1.0, 1.5, 1.2, 8.0, 9.0, 8.5];
    (dense, sparse, labels, responses)
}

/// Classifier fit and prediction agree identically across all four dense/CSR
/// combinations, including tree structures, importances and OOB accuracy.
#[test]
fn classifier_dense_and_sparse_fit_and_predict_are_equivalent() {
    let (dense, sparse, labels, _) = logical_fixture();
    let build = || {
        SKRandomForestClassifier::<f64>::build()
            .number_of_trees(20)
            .oob_score(true)
            .random_seed(42)
            .build()
            .expect("builds")
    };
    let dense_model = build().fit(&dense, labels.view()).expect("dense fit");
    let sparse_model = build().fit(sparse.view(), labels.view()).expect("sparse");

    assert!((0..20).all(|index| dense_model.tree(index) == sparse_model.tree(index)));
    assert_eq!(
        dense_model.feature_importances(),
        sparse_model.feature_importances()
    );
    assert_eq!(
        dense_model.oob_score().expect("oob"),
        sparse_model.oob_score().expect("oob")
    );

    let expected = dense_model.predict_probabilities(&dense).expect("dense");
    let dense_sparse = dense_model
        .predict_probabilities(sparse.view())
        .expect("sparse");
    let sparse_dense = sparse_model.predict_probabilities(&dense).expect("dense");
    let sparse_sparse = sparse_model
        .predict_probabilities(sparse.view())
        .expect("sparse");
    assert_eq!(
        (&expected, &expected, &expected),
        (&dense_sparse, &sparse_dense, &sparse_sparse)
    );
    assert_eq!(
        dense_model.predict_labels(&dense).expect("labels"),
        sparse_model.predict_labels(sparse.view()).expect("labels")
    );
}

/// Regressor fit and prediction agree identically across all four dense/CSR
/// combinations, including tree structures, importances and OOB R².
#[test]
fn regressor_dense_and_sparse_fit_and_predict_are_equivalent() {
    let (dense, sparse, _, responses) = logical_fixture();
    let build = || {
        SKRandomForestRegressor::<f64>::build()
            .number_of_trees(20)
            .oob_score(true)
            .random_seed(42)
            .build()
            .expect("builds")
    };
    let dense_model = build().fit(&dense, responses.view()).expect("dense fit");
    let sparse_model = build()
        .fit(sparse.view(), responses.view())
        .expect("sparse");

    assert!((0..20).all(|index| dense_model.tree(index) == sparse_model.tree(index)));
    assert_eq!(
        dense_model.feature_importances(),
        sparse_model.feature_importances()
    );
    assert_eq!(
        dense_model.oob_score().expect("oob"),
        sparse_model.oob_score().expect("oob")
    );

    let dense_to_dense = dense_model.predict(&dense).expect("dense");
    assert_eq!(
        dense_to_dense,
        dense_model.predict(sparse.view()).expect("sparse")
    );
    assert_eq!(dense_to_dense, sparse_model.predict(&dense).expect("dense"));
    assert_eq!(
        dense_to_dense,
        sparse_model.predict(sparse.view()).expect("sparse")
    );
}

/// An ultra-wide CSR design (`128 × 100_000`, where a dense matrix per tree
/// would exceed `100 MB`) fits and predicts from stored entries alone.
#[test]
fn ultra_wide_sparse_forest_fits_and_predicts_without_densifying() {
    let (rows, columns) = (128usize, 100_000usize);
    let mut indptr = Vec::with_capacity(rows + 1);
    let mut indices = Vec::with_capacity(rows * 2);
    let mut data = Vec::with_capacity(rows * 2);
    indptr.push(0);
    for row in 0..rows {
        indices.extend([0, 1_000 + (row * 701) % (columns - 1_000)]);
        data.extend([if row < 64 { 1.0 } else { 10.0 }, 2.0]);
        indptr.push(indices.len());
    }
    let sparse = CsMat::new((rows, columns), indptr, indices, data);
    let sampled_rows: Vec<usize> = (0..rows).collect();
    let gathered = sk_gather_sparse_rows(&sparse.view(), &sampled_rows);
    assert_eq!(
        (gathered.shape(), gathered.nnz()),
        ((rows, columns), rows * 2)
    );

    let labels = Array1::from_shape_fn(rows, |row| i64::from(row >= 64));
    let classifier = SKRandomForestClassifier::<f64>::build()
        .number_of_trees(16)
        .maximum_depth(3)
        .oob_score(true)
        .random_seed(3)
        .build()
        .expect("builds")
        .fit(sparse.view(), labels.view())
        .expect("wide sparse classifier fit");
    assert_eq!(
        classifier
            .predict_labels(sparse.view())
            .expect("pred")
            .len(),
        rows
    );

    let responses = Array1::from_shape_fn(rows, |row| if row < 64 { 1.0 } else { 10.0 });
    let regressor = SKRandomForestRegressor::<f64>::build()
        .number_of_trees(16)
        .maximum_depth(3)
        .oob_score(true)
        .random_seed(3)
        .build()
        .expect("builds")
        .fit(sparse.view(), responses.view())
        .expect("wide sparse regressor fit");
    assert_eq!(regressor.predict(sparse.view()).expect("pred").len(), rows);
}

/// Non-CSR (CSC) sparse views are rejected up front with
/// `UnsupportedRepresentation` on both fit and prediction.
#[test]
fn non_csr_sparse_storage_is_rejected_up_front() {
    let csc = CsMat::new_csc((4, 1), vec![0, 2], vec![1, 3], vec![1.0, 2.0]);
    let labels = array![0_i64, 0, 1, 1];
    let estimator = SKRandomForestClassifier::<f64>::build()
        .number_of_trees(4)
        .build()
        .expect("builds");
    assert!(matches!(
        estimator.fit(csc.view(), labels.view()),
        Err(SKError::UnsupportedRepresentation {
            representation: "csc",
            ..
        })
    ));
    let fitted = estimator
        .fit(&array![[0.0], [1.0], [2.0], [3.0]], labels.view())
        .expect("fit");
    assert!(matches!(
        fitted.predict_labels(csc.view()),
        Err(SKError::UnsupportedRepresentation {
            representation: "csc",
            ..
        })
    ));
}
