//! Companion tests for `Send + Sync` bounds and lock-free concurrent
//! prediction (task 4.2).

use std::sync::Arc;

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKClassifierPredictor, SKRegressorPredictor, SKSupervisedFit};
use sprs::CsMat;

use super::{
    SKRandomForestClassifier, SKRandomForestClassifierBuilder, SKRandomForestClassifierModel,
    SKRandomForestRegressor, SKRandomForestRegressorBuilder, SKRandomForestRegressorModel,
};

/// Estimators, builders and fitted models are `Send + Sync` for `f32` and `f64`.
#[test]
fn estimators_builders_and_models_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SKRandomForestClassifierBuilder<f64>>();
    assert_send_sync::<SKRandomForestRegressorBuilder<f64>>();
    assert_send_sync::<SKRandomForestClassifier<f64>>();
    assert_send_sync::<SKRandomForestRegressor<f64>>();
    assert_send_sync::<SKRandomForestClassifierModel<f64>>();
    assert_send_sync::<SKRandomForestRegressorModel<f64>>();
    assert_send_sync::<SKRandomForestClassifierModel<f32>>();
    assert_send_sync::<SKRandomForestRegressorModel<f32>>();
}

/// One fitted classification forest shared through `Arc` serves eight threads
/// concurrently on dense and CSR batches and matches single-thread output.
#[test]
fn shared_classifier_serves_concurrent_threads_deterministically() {
    let rows = 512usize;
    let dense = Arc::new(Array2::from_shape_fn((rows, 4), |(row, column)| {
        if (row + column) % 3 == 0 {
            ((row * 13 + column * 7) % 29) as f64
        } else {
            0.0
        }
    }));
    let sparse = Arc::new(sparse_from_dense(&dense));
    let labels = Array1::from_shape_fn(rows, |row| i64::from(dense[[row, 0]] > 10.0));

    let model = Arc::new(
        SKRandomForestClassifier::<f64>::build()
            .number_of_trees(24)
            .maximum_depth(5)
            .random_seed(101)
            .build()
            .expect("builds")
            .fit(dense.view(), labels.view())
            .expect("fit succeeds"),
    );

    let expected_probabilities = model
        .predict_probabilities(dense.view())
        .expect("probabilities");
    let expected_labels = model.predict_labels(dense.view()).expect("labels");

    let handles: Vec<_> = (0..8)
        .map(|thread_id| {
            let model = Arc::clone(&model);
            let dense = Arc::clone(&dense);
            let sparse = Arc::clone(&sparse);
            std::thread::spawn(move || {
                if thread_id % 2 == 0 {
                    (
                        model.predict_probabilities(dense.view()).expect("dense"),
                        model.predict_labels(dense.view()).expect("dense labels"),
                    )
                } else {
                    (
                        model.predict_probabilities(sparse.view()).expect("sparse"),
                        model.predict_labels(sparse.view()).expect("sparse labels"),
                    )
                }
            })
        })
        .collect();

    for handle in handles {
        let (probabilities, labels) = handle.join().expect("thread joins");
        assert_eq!(probabilities, expected_probabilities);
        assert_eq!(labels, expected_labels);
    }
}

/// One fitted regression forest shared through `Arc` serves eight threads
/// concurrently on dense and CSR batches and matches single-thread output.
#[test]
fn shared_regressor_serves_concurrent_threads_deterministically() {
    let rows = 512usize;
    let dense = Arc::new(Array2::from_shape_fn((rows, 4), |(row, column)| {
        if (row + column) % 2 == 0 {
            ((row * 11 + column * 5) % 23) as f64
        } else {
            0.0
        }
    }));
    let sparse = Arc::new(sparse_from_dense(&dense));
    let responses = Array1::from_shape_fn(rows, |row| 2.0 * dense[[row, 0]] + dense[[row, 2]]);

    let model = Arc::new(
        SKRandomForestRegressor::<f64>::build()
            .number_of_trees(24)
            .maximum_depth(5)
            .random_seed(202)
            .build()
            .expect("builds")
            .fit(dense.view(), responses.view())
            .expect("fit succeeds"),
    );

    let expected = model.predict(dense.view()).expect("predict");
    let handles: Vec<_> = (0..8)
        .map(|thread_id| {
            let model = Arc::clone(&model);
            let dense = Arc::clone(&dense);
            let sparse = Arc::clone(&sparse);
            std::thread::spawn(move || {
                if thread_id % 2 == 0 {
                    model.predict(dense.view()).expect("dense")
                } else {
                    model.predict(sparse.view()).expect("sparse")
                }
            })
        })
        .collect();

    for handle in handles {
        assert_eq!(handle.join().expect("thread joins"), expected);
    }
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
