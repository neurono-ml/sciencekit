//! Large-data (`12 000` rows) acceptance tests for generic bagging: tree and
//! non-tree bases, feature bootstrap, OOB on/off, dense/CSR parity, thread-
//! count invariance and `sciencekit_metrics` evaluation.

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit};
use sciencekit_linear_model::ridge::SKRidge;
use sciencekit_metrics::classification::{SKAveraging, sk_accuracy_score, sk_f1_score};
use sciencekit_metrics::regression::{sk_mean_absolute_error, sk_mean_squared_error, sk_r2_score};
use sciencekit_tree::{SKDecisionTreeClassifier, SKDecisionTreeRegressor, SKMaximumFeatures};
use sprs::CsMat;

use crate::{SKBaggingClassifier, SKBaggingRegressor, SKMaximumSamples};

const LARGE_ROWS: usize = 12_000;
const COLUMNS: usize = 6;

fn sparse_from_dense(dense: &Array2<f64>) -> CsMat<f64> {
    let mut indptr = Vec::with_capacity(dense.nrows() + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();
    indptr.push(0);
    dense.rows().into_iter().for_each(|row| {
        row.iter()
            .enumerate()
            .filter(|(_, value)| **value != 0.0)
            .for_each(|(column, &value)| {
                indices.push(column);
                data.push(value);
            });
        indptr.push(indices.len());
    });
    CsMat::new((dense.nrows(), dense.ncols()), indptr, indices, data)
}

fn large_classification_dataset() -> (Array2<f64>, CsMat<f64>, Array1<i64>) {
    let dense = Array2::from_shape_fn((LARGE_ROWS, COLUMNS), |(row, column)| {
        let raw = ((row * 17 + column * 31) % 101) as f64 / 50.0 - 1.0;
        if (row + column) % 3 == 0 { 0.0 } else { raw }
    });
    let sparse = sparse_from_dense(&dense);
    let labels = Array1::from_shape_fn(LARGE_ROWS, |row| {
        let score = dense[[row, 0]] + dense[[row, 1]] - dense[[row, 2]] + 0.5 * dense[[row, 3]];
        i64::from(score > 0.0)
    });
    (dense, sparse, labels)
}

fn large_regression_dataset() -> (Array2<f64>, CsMat<f64>, Array1<f64>) {
    let dense = Array2::from_shape_fn((LARGE_ROWS, COLUMNS), |(row, column)| {
        let raw = ((row * 13 + column * 29) % 97) as f64 / 48.5 - 1.0;
        if (row + column) % 4 == 0 { 0.0 } else { raw }
    });
    let sparse = sparse_from_dense(&dense);
    let targets = Array1::from_shape_fn(LARGE_ROWS, |row| {
        3.0 * dense[[row, 0]] - 2.0 * dense[[row, 1]] + 0.5 * dense[[row, 2]]
    });
    (dense, sparse, targets)
}

#[test]
fn large_dense_and_sparse_bagging_classifier_with_feature_bootstrap_and_metrics() {
    let (dense, sparse, labels) = large_classification_dataset();
    let base_tree = SKDecisionTreeClassifier::<f64>::build()
        .maximum_depth(6)
        .build()
        .unwrap();

    let dense_oob = SKBaggingClassifier::build(base_tree.clone())
        .number_of_estimators(12)
        .maximum_samples(SKMaximumSamples::Fraction(0.8))
        .maximum_features(SKMaximumFeatures::Count(5))
        .bootstrap_features(true)
        .oob_score(true)
        .random_seed(42)
        .build()
        .unwrap()
        .fit(&dense, labels.view())
        .unwrap();

    let sparse_no_oob = SKBaggingClassifier::build(base_tree)
        .number_of_estimators(12)
        .maximum_samples(SKMaximumSamples::Fraction(0.8))
        .maximum_features(SKMaximumFeatures::Count(5))
        .bootstrap_features(true)
        .oob_score(false)
        .random_seed(42)
        .build()
        .unwrap()
        .fit(sparse.view(), labels.view())
        .unwrap();

    let dense_predictions = dense_oob.predict_labels(&dense).unwrap();
    let sparse_predictions = sparse_no_oob.predict_labels(sparse.view()).unwrap();
    assert_eq!(dense_predictions, sparse_predictions);
    assert!(sparse_no_oob.oob_score().is_err());

    let truth: Vec<i64> = labels
        .iter()
        .map(|label| dense_oob.classes().index_of(&label.to_string()).unwrap() as i64)
        .collect();
    let predicted_slice = dense_predictions.as_slice().unwrap();
    let accuracy: f64 = sk_accuracy_score(&truth, predicted_slice).unwrap();
    let macro_f1: f64 = sk_f1_score(&truth, predicted_slice, None, SKAveraging::Macro).unwrap();
    let oob_accuracy = dense_oob.oob_score().unwrap();

    assert!(accuracy > 0.80, "accuracy {accuracy}");
    assert!(macro_f1 > 0.80, "macro_f1 {macro_f1}");
    assert!(oob_accuracy > 0.78, "oob_accuracy {oob_accuracy}");
}

#[test]
fn large_non_tree_and_tree_bagging_regressors_thread_invariance_and_metrics() {
    let (dense, sparse, targets) = large_regression_dataset();

    // Non-tree sparse-capable Ridge regressor over 12 000 CSR rows.
    let ridge_base = SKRidge::<f64>::build().alpha(1.0).build().unwrap();
    let ridge_bagging = SKBaggingRegressor::build(ridge_base)
        .number_of_estimators(8)
        .oob_score(true)
        .random_seed(2026)
        .build()
        .unwrap()
        .fit(sparse.view(), targets.view())
        .unwrap();
    let ridge_predictions = ridge_bagging.predict(sparse.view()).unwrap();
    let mean_squared = sk_mean_squared_error(targets.view(), ridge_predictions.view()).unwrap();
    let mean_absolute = sk_mean_absolute_error(targets.view(), ridge_predictions.view()).unwrap();
    let r2 = sk_r2_score(targets.view(), ridge_predictions.view()).unwrap();
    assert!(mean_squared < 0.05, "mean_squared {mean_squared}");
    assert!(mean_absolute < 0.20, "mean_absolute {mean_absolute}");
    assert!(r2 > 0.98, "r2 {r2}");
    assert!(ridge_bagging.oob_score().unwrap() > 0.98);

    // Thread-count invariance across 1-thread and 4-thread rayon pools on 12 000 rows.
    let tree_base = SKDecisionTreeRegressor::<f64>::build()
        .maximum_depth(5)
        .build()
        .unwrap();
    let estimator = SKBaggingRegressor::build(tree_base)
        .number_of_estimators(10)
        .maximum_features(SKMaximumFeatures::Count(4))
        .oob_score(true)
        .random_seed(2026)
        .build()
        .unwrap();

    let run_with_threads = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let model = estimator.fit(&dense, targets.view()).unwrap();
                let predictions = model.predict(&dense).unwrap();
                (model, predictions)
            })
    };

    let (one_model, one_predictions) = run_with_threads(1);
    let (four_model, four_predictions) = run_with_threads(4);
    assert_eq!(one_model.feature_subsets(), four_model.feature_subsets());
    assert_eq!(
        one_model.oob_score().unwrap(),
        four_model.oob_score().unwrap()
    );
    assert_eq!(one_predictions, four_predictions);
}
