//! Dense/sparse parity tests for the histogram boosters (task 5.1).

use ndarray::Array2;
use sciencekit_common::{SKBuilder, SKRegressorPredictor, SKSupervisedFit};
use sprs::CsMat;

use super::SKHistGradientBoostingRegressor;

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

fn data() -> (Array2<f64>, ndarray::Array1<f64>) {
    let rows = 90usize;
    let features = Array2::from_shape_fn((rows, 3), |(row, column)| {
        if (row + column) % 4 == 0 {
            0.0
        } else {
            ((row * 7 + column * 5) % 19) as f64 / 3.0
        }
    });
    let targets = ndarray::Array1::from_shape_fn(rows, |row| {
        features[[row, 0]] - 0.5 * features[[row, 1]] + features[[row, 2]]
    });
    (features, targets)
}

fn build() -> SKHistGradientBoostingRegressor<f64> {
    SKHistGradientBoostingRegressor::<f64>::build()
        .number_of_estimators(12)
        .learning_rate(0.3)
        .maximum_leaf_nodes(6)
        .minimum_samples_leaf(1)
        .maximum_bins(16)
        .without_early_stopping()
        .build()
        .expect("build")
}

#[test]
fn hist_representation_sparse_fit_equals_dense_fit() {
    let (dense, targets) = data();
    let sparse = sparse_from_dense(&dense);
    let dense_model = build()
        .fit(dense.view(), targets.view())
        .expect("dense fit");
    let sparse_model = build()
        .fit(sparse.view(), targets.view())
        .expect("sparse fit");
    let dense_predictions = dense_model.predict(dense.view()).expect("dense predict");
    let sparse_predictions = sparse_model.predict(sparse.view()).expect("sparse predict");
    dense_predictions
        .iter()
        .zip(sparse_predictions.iter())
        .for_each(|(dense_value, sparse_value)| {
            assert!((dense_value - sparse_value).abs() < 1e-12);
        });
}

#[test]
fn hist_representation_dense_and_sparse_queries_agree() {
    let (dense, targets) = data();
    let model = build().fit(dense.view(), targets.view()).expect("fit");
    let dense_predictions = model.predict(dense.view()).expect("dense predict");
    let sparse = sparse_from_dense(&dense);
    let sparse_predictions = model.predict(sparse.view()).expect("sparse predict");
    dense_predictions
        .iter()
        .zip(sparse_predictions.iter())
        .for_each(|(dense_value, sparse_value)| {
            assert!((dense_value - sparse_value).abs() < 1e-12);
        });
}
