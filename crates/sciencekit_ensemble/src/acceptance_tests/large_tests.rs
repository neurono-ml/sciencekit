//! Large-data acceptance tests (`>= 20 000` rows): classifier and regressor,
//! OOB on/off, dense/CSR, thread-count invariance and metrics (task 4.3).

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit};
use sciencekit_metrics::classification::{SKAveraging, sk_accuracy_score, sk_f1_score};
use sciencekit_metrics::regression::{sk_mean_absolute_error, sk_mean_squared_error, sk_r2_score};
use sprs::CsMat;

use crate::{SKRandomForestClassifier, SKRandomForestRegressor};

const ROWS: usize = 20_000;
const COLUMNS: usize = 5;

fn classification_dataset() -> (Array2<f64>, CsMat<f64>, Array1<i64>) {
    let dense = Array2::from_shape_fn((ROWS, COLUMNS), |(row, column)| {
        let raw = ((row * 17 + column * 31) % 101) as f64 / 50.0 - 1.0;
        if (row + column) % 3 == 0 { 0.0 } else { raw }
    });
    let sparse = sparse_from_dense(&dense);
    let labels = Array1::from_shape_fn(ROWS, |row| {
        let score = dense[[row, 0]] + dense[[row, 1]] - dense[[row, 2]] + 0.5 * dense[[row, 3]];
        i64::from(score > 0.0)
    });
    (dense, sparse, labels)
}

fn regression_dataset() -> (Array2<f64>, CsMat<f64>, Array1<f64>) {
    let dense = Array2::from_shape_fn((ROWS, COLUMNS), |(row, column)| {
        let raw = ((row * 13 + column * 29) % 97) as f64 / 48.5 - 1.0;
        if (row + column) % 4 == 0 { 0.0 } else { raw }
    });
    let sparse = sparse_from_dense(&dense);
    let responses = Array1::from_shape_fn(ROWS, |row| {
        3.0 * dense[[row, 0]] - 2.0 * dense[[row, 1]] + 0.5 * dense[[row, 2]]
    });
    (dense, sparse, responses)
}

fn sparse_from_dense(dense: &Array2<f64>) -> CsMat<f64> {
    let mut indptr = Vec::with_capacity(dense.nrows() + 1);
    let mut indices = Vec::new();
    let mut data = Vec::new();
    indptr.push(0);
    for row in dense.rows() {
        for (column, &value) in row.iter().enumerate() {
            if value != 0.0 {
                indices.push(column);
                data.push(value);
            }
        }
        indptr.push(indices.len());
    }
    CsMat::new((dense.nrows(), dense.ncols()), indptr, indices, data)
}

/// 20 000-row classification fits densely and sparsely with OOB on and off and
/// achieves high accuracy and macro-F1 through `sciencekit_metrics`.
#[test]
fn large_dense_and_sparse_classification_fits_with_oob_and_metrics() {
    let (dense, sparse, labels) = classification_dataset();
    let dense_oob = SKRandomForestClassifier::<f64>::build()
        .number_of_trees(16)
        .maximum_depth(6)
        .oob_score(true)
        .random_seed(42)
        .build()
        .expect("builds")
        .fit(&dense, labels.view())
        .expect("dense fit");
    let sparse_no_oob = SKRandomForestClassifier::<f64>::build()
        .number_of_trees(16)
        .maximum_depth(6)
        .oob_score(false)
        .random_seed(42)
        .build()
        .expect("builds")
        .fit(sparse.view(), labels.view())
        .expect("sparse fit");

    let truth: Vec<i64> = labels
        .iter()
        .map(|label| {
            dense_oob
                .classes()
                .index_of(&label.to_string())
                .expect("class") as i64
        })
        .collect();
    let dense_pred = dense_oob.predict_labels(&dense).expect("dense predict");
    let sparse_pred = sparse_no_oob
        .predict_labels(sparse.view())
        .expect("sparse predict");
    assert_eq!(dense_pred, sparse_pred);
    assert!(sparse_no_oob.oob_score().is_err());

    let accuracy: f64 =
        sk_accuracy_score(&truth, dense_pred.as_slice().expect("slice")).expect("accuracy");
    let macro_f1: f64 = sk_f1_score(
        &truth,
        dense_pred.as_slice().expect("slice"),
        None,
        SKAveraging::Macro,
    )
    .expect("f1");
    let oob_accuracy = dense_oob.oob_score().expect("oob");
    assert!(accuracy > 0.85, "accuracy {accuracy}");
    assert!(macro_f1 > 0.85, "macro_f1 {macro_f1}");
    assert!(oob_accuracy > 0.82, "oob_accuracy {oob_accuracy}");
}

/// 20 000-row regression fits densely and sparsely with OOB on and off and
/// produces small MSE/MAE and high R² through `sciencekit_metrics`.
#[test]
fn large_dense_and_sparse_regression_fits_with_oob_and_metrics() {
    let (dense, sparse, responses) = regression_dataset();
    let dense_oob = SKRandomForestRegressor::<f64>::build()
        .number_of_trees(16)
        .maximum_depth(6)
        .oob_score(true)
        .random_seed(42)
        .build()
        .expect("builds")
        .fit(&dense, responses.view())
        .expect("dense fit");
    let sparse_no_oob = SKRandomForestRegressor::<f64>::build()
        .number_of_trees(16)
        .maximum_depth(6)
        .oob_score(false)
        .random_seed(42)
        .build()
        .expect("builds")
        .fit(sparse.view(), responses.view())
        .expect("sparse fit");

    let dense_pred = dense_oob.predict(&dense).expect("dense predict");
    let sparse_pred = sparse_no_oob
        .predict(sparse.view())
        .expect("sparse predict");
    assert_eq!(dense_pred, sparse_pred);
    assert!(sparse_no_oob.oob_score().is_err());

    let mse = sk_mean_squared_error(responses.view(), dense_pred.view()).expect("mse");
    let mae = sk_mean_absolute_error(responses.view(), dense_pred.view()).expect("mae");
    let r2 = sk_r2_score(responses.view(), dense_pred.view()).expect("r2");
    let oob_r2 = dense_oob.oob_score().expect("oob r2");
    assert!(mse < 0.25, "mse {mse}");
    assert!(mae < 0.40, "mae {mae}");
    assert!(r2 > 0.90, "r2 {r2}");
    assert!(oob_r2 > 0.85, "oob_r2 {oob_r2}");
}

/// Fitting and predicting 20 000 rows on 1-thread and 4-thread rayon pools
/// yields identical trees, OOB scores, importances and predictions.
#[test]
fn large_forest_is_invariant_across_rayon_thread_counts() {
    let (dense, _, responses) = regression_dataset();
    let estimator = SKRandomForestRegressor::<f64>::build()
        .number_of_trees(12)
        .maximum_depth(5)
        .oob_score(true)
        .random_seed(2026)
        .build()
        .expect("builds");

    let run_with_threads = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .expect("pool")
            .install(|| {
                let model = estimator.fit(&dense, responses.view()).expect("fit");
                let predictions = model.predict(&dense).expect("predict");
                (model, predictions)
            })
    };

    let (model_1, pred_1) = run_with_threads(1);
    let (model_4, pred_4) = run_with_threads(4);
    for index in 0..12 {
        assert_eq!(model_1.tree(index), model_4.tree(index));
    }
    assert_eq!(model_1.feature_importances(), model_4.feature_importances());
    assert_eq!(
        model_1.oob_score().expect("oob 1"),
        model_4.oob_score().expect("oob 4")
    );
    assert_eq!(pred_1, pred_4);
}
