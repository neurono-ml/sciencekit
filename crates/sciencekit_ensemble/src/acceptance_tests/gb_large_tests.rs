//! Large-data (`>= 20 000` rows) acceptance tests for gradient boosting:
//! binary and multiclass classification, all regression losses, subsampling,
//! early stopping, dense/CSR parity and metrics via `sciencekit_metrics`
//! (task 7.1).

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit, sk_canonicalize_labels,
};
use sciencekit_metrics::classification::{SKAveraging, sk_accuracy_score, sk_f1_score};
use sciencekit_metrics::ranking::sk_log_loss;
use sciencekit_metrics::regression::{sk_mean_absolute_error, sk_mean_squared_error, sk_r2_score};
use sciencekit_tree::SKMaximumFeatures;
use sprs::CsMat;

use crate::{
    SKGradientBoostingClassifier, SKGradientBoostingRegressionLoss, SKGradientBoostingRegressor,
};

const ROWS: usize = 20_000;
const COLUMNS: usize = 5;

fn sparse_from_dense(dense: &Array2<f64>) -> CsMat<f64> {
    let mut indptr = Vec::with_capacity(dense.nrows() + 1);
    let (mut indices, mut data) = (Vec::new(), Vec::new());
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

fn large_design() -> (Array2<f64>, CsMat<f64>) {
    let dense = Array2::from_shape_fn((ROWS, COLUMNS), |(row, column)| {
        let raw = ((row * 17 + column * 31) % 101) as f64 / 50.0 - 1.0;
        if (row + column) % 3 == 0 { 0.0 } else { raw }
    });
    let sparse = sparse_from_dense(&dense);
    (dense, sparse)
}

/// 20 000-row binary and 3-class gradient boosting fit densely and sparsely
/// with stochastic subsampling and early stopping, producing strong accuracy,
/// macro-F1 and log-loss through `sciencekit_metrics`.
#[test]
fn large_binary_and_multiclass_gradient_boosting_with_metrics() {
    let (dense, sparse) = large_design();
    let binary_labels = Array1::from_shape_fn(ROWS, |row| {
        let score = dense[[row, 0]] + dense[[row, 1]] - dense[[row, 2]] + 0.5 * dense[[row, 3]];
        i64::from(score > 0.0)
    });
    let build_binary = || {
        SKGradientBoostingClassifier::<f64>::build()
            .number_of_estimators(16)
            .learning_rate(0.25)
            .maximum_depth(4)
            .subsample(0.8)
            .maximum_features(SKMaximumFeatures::Sqrt)
            .n_iter_no_change(4)
            .random_seed(42)
            .build()
            .expect("builds")
    };
    let dense_binary = build_binary()
        .fit(&dense, binary_labels.view())
        .expect("dense binary fit");
    let sparse_binary = build_binary()
        .fit(sparse.view(), binary_labels.view())
        .expect("sparse binary fit");

    let binary_probs = dense_binary.predict_probabilities(&dense).expect("probs");
    let sparse_probs = sparse_binary
        .predict_probabilities(sparse.view())
        .expect("sparse probs");
    assert_eq!(binary_probs, sparse_probs);

    let binary_preds = dense_binary.predict_labels(&dense).expect("labels");
    let binary_truth = binary_labels.as_slice().expect("slice");
    let binary_pred_slice = binary_preds.as_slice().expect("slice");
    let accuracy: f64 = sk_accuracy_score(binary_truth, binary_pred_slice).expect("accuracy");
    let macro_f1: f64 =
        sk_f1_score(binary_truth, binary_pred_slice, None, SKAveraging::Macro).expect("f1");
    let loss = sk_log_loss(binary_truth, binary_probs.view(), None).expect("log_loss");
    assert!(accuracy > 0.85, "binary accuracy {accuracy}");
    assert!(macro_f1 > 0.85, "binary macro_f1 {macro_f1}");
    assert!(loss < 0.40, "binary log_loss {loss}");

    let multiclass_labels = Array1::from_shape_fn(ROWS, |row| {
        let score = 2.0 * dense[[row, 0]] - 1.5 * dense[[row, 1]] + dense[[row, 2]];
        if score < -0.6 {
            0_i64
        } else if score < 0.6 {
            1
        } else {
            2
        }
    });
    let multiclass_model = SKGradientBoostingClassifier::<f64>::build()
        .number_of_estimators(14)
        .learning_rate(0.25)
        .maximum_depth(4)
        .subsample(0.8)
        .random_seed(7)
        .build()
        .expect("builds")
        .fit(sparse.view(), multiclass_labels.view())
        .expect("multiclass sparse fit");
    assert_eq!(multiclass_model.trees_per_stage(), 3);
    let multiclass_preds = multiclass_model
        .predict_labels(sparse.view())
        .expect("multiclass predict");
    let mc_truth = multiclass_labels.as_slice().expect("slice");
    let mc_pred = multiclass_preds.as_slice().expect("slice");
    let owned_truth: Vec<String> = mc_truth.iter().map(ToString::to_string).collect();
    let borrowed_truth: Vec<&str> = owned_truth.iter().map(String::as_str).collect();
    let (canonical_truth, _) = sk_canonicalize_labels(&borrowed_truth);
    let canonical_truth: Vec<i64> = canonical_truth
        .into_iter()
        .map(|index| index as i64)
        .collect();
    let mc_accuracy: f64 = sk_accuracy_score(&canonical_truth, mc_pred).expect("mc accuracy");
    let mc_f1: f64 =
        sk_f1_score(&canonical_truth, mc_pred, None, SKAveraging::Macro).expect("mc f1");
    assert!(mc_accuracy > 0.82, "multiclass accuracy {mc_accuracy}");
    assert!(mc_f1 > 0.82, "multiclass macro_f1 {mc_f1}");
}

/// 20 000-row regression boosting covers `SquaredError`, `AbsoluteError` and
/// `Huber`, dense/CSR parity, subsampling, early stopping and regression
/// metrics (`MSE`, `MAE`, `R²`).
#[test]
fn large_regression_gradient_boosting_all_losses_early_stop_and_metrics() {
    let (dense, sparse) = large_design();
    let responses = Array1::from_shape_fn(ROWS, |row| {
        3.0 * dense[[row, 0]] - 2.0 * dense[[row, 1]] + 0.5 * dense[[row, 2]]
    });

    for loss in [
        SKGradientBoostingRegressionLoss::SquaredError,
        SKGradientBoostingRegressionLoss::AbsoluteError,
        SKGradientBoostingRegressionLoss::Huber,
    ] {
        let build_regressor = || {
            SKGradientBoostingRegressor::<f64>::build()
                .number_of_estimators(24)
                .learning_rate(0.25)
                .loss(loss)
                .huber_delta(1.5)
                .maximum_depth(4)
                .subsample(0.8)
                .random_seed(2026)
                .build()
                .expect("builds")
        };
        let dense_model = build_regressor()
            .fit(&dense, responses.view())
            .expect("dense fit");
        let sparse_model = build_regressor()
            .fit(sparse.view(), responses.view())
            .expect("sparse fit");

        let dense_preds = dense_model.predict(&dense).expect("dense predict");
        let sparse_preds = sparse_model.predict(sparse.view()).expect("sparse predict");
        assert_eq!(dense_preds, sparse_preds);

        let mse = sk_mean_squared_error(responses.view(), dense_preds.view()).expect("mse");
        let mae = sk_mean_absolute_error(responses.view(), dense_preds.view()).expect("mae");
        let r2 = sk_r2_score(responses.view(), dense_preds.view()).expect("r2");
        assert!(mse < 0.45, "loss {loss:?} mse {mse}");
        assert!(mae < 0.50, "loss {loss:?} mae {mae}");
        assert!(r2 > 0.85, "loss {loss:?} r2 {r2}");
    }

    let step_targets =
        Array1::from_shape_fn(ROWS, |row| if dense[[row, 0]] <= 0.0 { -4.0 } else { 4.0 });
    let early_stopped = SKGradientBoostingRegressor::<f64>::build()
        .number_of_estimators(30)
        .learning_rate(1.0)
        .maximum_depth(1)
        .n_iter_no_change(3)
        .validation_fraction(0.1)
        .random_seed(9)
        .build()
        .expect("builds")
        .fit(sparse.view(), step_targets.view())
        .expect("early-stop fit");
    assert!(early_stopped.number_of_stages() < 30);
}
