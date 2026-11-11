//! Companion tests for dense/sparse representation parity and non-densifying
//! sparse fitting (task 3.3).

use ndarray::{Array1, array};
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKError, SKRegressorPredictor, SKSupervisedFit,
};
use sprs::CsMat;

use crate::bootstrap::sk_gather_sparse_rows;

use super::{SKRandomForestClassifier, SKRandomForestRegressor};

fn logical_fixture() -> (
    ndarray::Array2<f64>,
    CsMat<f64>,
    Array1<i64>,
    Array1<f64>,
) {
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
    let builder = || {
        SKRandomForestClassifier::<f64>::build()
            .number_of_trees(20)
            .oob_score(true)
            .random_seed(42)
            .build()
            .expect("builds")
    };
    let dense_model = builder().fit(&dense, labels.view()).expect("dense fit");
    let sparse_model = builder()
        .fit(sparse.view(), labels.view())
        .expect("sparse fit");

    for tree_index in 0..20 {
        assert_eq!(dense_model.tree(tree_index), sparse_model.tree(tree_index));
    }
    assert_eq!(
        dense_model.feature_importances(),
        sparse_model.feature_importances()
    );
    assert_eq!(
        dense_model.oob_score().expect("dense oob"),
        sparse_model.oob_score().expect("sparse oob")
    );

    let dd = dense_model
        .predict_probabilities(&dense)
        .expect("dense->dense");
    let ds = dense_model
        .predict_probabilities(sparse.view())
        .expect("dense->sparse");
    let sd = sparse_model
        .predict_probabilities(&dense)
        .expect("sparse->dense");
    let ss = sparse_model
        .predict_probabilities(sparse.view())
        .expect("sparse->sparse");
    assert_eq!(dd, ds);
    assert_eq!(dd, sd);
    assert_eq!(dd, ss);
    assert_eq!(
        dense_model.predict_labels(&dense).expect("labels"),
        sparse_model
            .predict_labels(sparse.view())
            .expect("sparse labels")
    );
}

/// Regressor fit and prediction agree identically across all four dense/CSR
/// combinations, including tree structures, importances and OOB R².
#[test]
fn regressor_dense_and_sparse_fit_and_predict_are_equivalent() {
    let (dense, sparse, _, responses) = logical_fixture();
    let builder = || {
        SKRandomForestRegressor::<f64>::build()
            .number_of_trees(20)
            .oob_score(true)
            .random_seed(42)
            .build()
            .expect("builds")
    };
    let dense_model = builder().fit(&dense, responses.view()).expect("dense fit");
    let sparse_model = builder()
        .fit(sparse.view(), responses.view())
        .expect("sparse fit");

    for tree_index in 0..20 {
        assert_eq!(dense_model.tree(tree_index), sparse_model.tree(tree_index));
    }
    assert_eq!(
        dense_model.feature_importances(),
        sparse_model.feature_importances()
    );
    assert_eq!(
        dense_model.oob_score().expect("dense oob"),
        sparse_model.oob_score().expect("sparse oob")
    );

    let dd = dense_model.predict(&dense).expect("dense->dense");
    let ds = dense_model.predict(sparse.view()).expect("dense->sparse");
    let sd = sparse_model.predict(&dense).expect("sparse->dense");
    let ss = sparse_model.predict(sparse.view()).expect("sparse->sparse");
    assert_eq!(dd, ds);
    assert_eq!(dd, sd);
    assert_eq!(dd, ss);
}

/// An ultra-wide CSR design (`128 × 100_000`, where a dense matrix per tree
/// would exceed `100 MB` and `1.6 GB` across 16 trees) fits and predicts from
/// stored entries alone without densifying.
#[test]
fn ultra_wide_sparse_forest_fits_and_predicts_without_densifying() {
    let rows = 128usize;
    let columns = 100_000usize;
    let mut indptr = Vec::with_capacity(rows + 1);
    let mut indices = Vec::with_capacity(rows * 2);
    let mut data = Vec::with_capacity(rows * 2);
    indptr.push(0);
    for row in 0..rows {
        indices.push(0);
        data.push(if row < 64 { 1.0 } else { 10.0 });
        let high_column = 1_000 + (row * 701) % (columns - 1_000);
        indices.push(high_column);
        data.push(2.0);
        indptr.push(indices.len());
    }
    let sparse = CsMat::new((rows, columns), indptr, indices, data);
    assert_eq!(sparse.nnz(), rows * 2);

    // Direct check on bootstrap CSR gathering: stored entries stay O(nnz),
    // never O(rows * columns).
    let sampled_rows: Vec<usize> = (0..rows).collect();
    let gathered = sk_gather_sparse_rows(&sparse.view(), &sampled_rows);
    assert_eq!(gathered.shape(), (rows, columns));
    assert_eq!(gathered.nnz(), rows * 2);

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
    assert_eq!(classifier.feature_count(), columns);
    let predicted = classifier
        .predict_labels(sparse.view())
        .expect("wide sparse predict");
    assert_eq!(predicted.len(), rows);

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
    let reg_pred = regressor
        .predict(sparse.view())
        .expect("wide sparse reg predict");
    assert_eq!(reg_pred.len(), rows);
}

/// Non-CSR (CSC) sparse views are rejected up front with
/// `UnsupportedRepresentation` on both fit and prediction instead of panicking.
#[test]
fn non_csr_sparse_storage_is_rejected_up_front() {
    let csc = CsMat::new_csc((4, 1), vec![0, 2], vec![1, 3], vec![1.0, 2.0]);
    let labels = array![0_i64, 0, 1, 1];
    let estimator = SKRandomForestClassifier::<f64>::build()
        .number_of_trees(4)
        .build()
        .expect("builds");
    let fit_error = estimator
        .fit(csc.view(), labels.view())
        .expect_err("CSC fit must fail");
    assert!(matches!(
        fit_error,
        SKError::UnsupportedRepresentation {
            representation: "csc",
            ..
        }
    ));

    let dense = array![[0.0], [1.0], [2.0], [3.0]];
    let fitted = estimator.fit(&dense, labels.view()).expect("dense fit");
    let predict_error = fitted
        .predict_labels(csc.view())
        .expect_err("CSC predict must fail");
    assert!(matches!(
        predict_error,
        SKError::UnsupportedRepresentation {
            representation: "csc",
            ..
        }
    ));
}
