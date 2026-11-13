//! Companion tests for MiniBatchKMeans input and configuration validation
//! (spec `mini-batch-k-means`, task 4.3).

use std::collections::VecDeque;

use ndarray::{Array2, array};
use sciencekit_common::{
    SKBuilder, SKClustererPredictor, SKDataBatch, SKError, SKLazySource, SKUnsupervisedFit,
};
use sprs::{CsMat, TriMat};

use super::SKMiniBatchKMeans;

struct VecSource(VecDeque<SKDataBatch<f64>>);

impl SKLazySource<f64> for VecSource {
    type Error = SKError;

    fn next_batch(&mut self) -> Result<Option<SKDataBatch<f64>>, Self::Error> {
        Ok(self.0.pop_front())
    }
}

fn csr(entries: &[(usize, usize, f64)], shape: (usize, usize)) -> CsMat<f64> {
    let mut triplet = TriMat::new(shape);
    entries
        .iter()
        .for_each(|&(row, column, value)| triplet.add_triplet(row, column, value));
    triplet.to_csr()
}

#[test]
fn minibatch_validation_rejects_empty_inputs_across_operations() {
    let estimator = SKMiniBatchKMeans::<f64>::build()
        .number_of_clusters(2)
        .build()
        .unwrap();

    for empty in [Array2::<f64>::zeros((0, 2)), Array2::<f64>::zeros((3, 0))] {
        assert!(matches!(
            estimator.fit(&empty).unwrap_err(),
            SKError::EmptyInput { operation: "fit" }
        ));
        assert!(matches!(
            estimator.fit_predict(&empty).unwrap_err(),
            SKError::EmptyInput {
                operation: "fit_predict"
            }
        ));
        assert!(matches!(
            estimator.fit_batch(&empty).unwrap_err(),
            SKError::EmptyInput {
                operation: "partial_fit"
            }
        ));
    }

    let mut no_batches = VecSource(VecDeque::new());
    assert!(matches!(
        estimator.fit_streaming(&mut no_batches).unwrap_err(),
        SKError::EmptyInput {
            operation: "fit_streaming"
        }
    ));
    let mut zero_row_batch = VecSource(VecDeque::from([SKDataBatch::new_dense(
        Array2::<f64>::zeros((0, 2)),
        0,
        true,
    )]));
    assert!(matches!(
        estimator.fit_streaming(&mut zero_row_batch).unwrap_err(),
        SKError::EmptyInput {
            operation: "fit_streaming"
        }
    ));

    let mut fitted = estimator
        .fit(&array![[0.0_f64, 0.0], [0.2, 0.1], [10.0, 10.0]])
        .unwrap();
    let empty_query = Array2::<f64>::zeros((0, 2));
    assert!(matches!(
        fitted.partial_fit(&empty_query).unwrap_err(),
        SKError::EmptyInput {
            operation: "partial_fit"
        }
    ));
    assert!(matches!(
        fitted.predict_clusters(&empty_query).unwrap_err(),
        SKError::EmptyInput {
            operation: "predict"
        }
    ));
    assert!(matches!(
        fitted.transform(&empty_query).unwrap_err(),
        SKError::EmptyInput {
            operation: "transform"
        }
    ));
}

#[test]
fn minibatch_validation_rejects_small_batches_and_preserves_existing_model_state() {
    let estimator = SKMiniBatchKMeans::<f64>::build()
        .number_of_clusters(4)
        .build()
        .unwrap();
    let three_rows = array![[0.0_f64, 0.0], [1.0, 1.0], [2.0, 2.0]];

    assert!(matches!(
        estimator.fit(&three_rows).unwrap_err(),
        SKError::InvalidHyperparameter { .. }
    ));
    assert!(matches!(
        estimator.fit_batch(&three_rows).unwrap_err(),
        SKError::InvalidHyperparameter { .. }
    ));
    let mut small_stream = VecSource(VecDeque::from([SKDataBatch::new_dense(
        three_rows.clone(),
        0,
        true,
    )]));
    assert!(matches!(
        estimator.fit_streaming(&mut small_stream).unwrap_err(),
        SKError::InvalidHyperparameter { .. }
    ));

    let initial = array![[0.0_f64, 0.0], [1.0, 0.0], [10.0, 10.0], [20.0, 20.0]];
    let mut model = estimator.fit_batch(&initial).unwrap();
    let snapshot_centers = model.cluster_centers().to_owned();
    let snapshot_counts = model.counts().to_owned();

    assert!(matches!(
        model.partial_fit(&three_rows).unwrap_err(),
        SKError::InvalidHyperparameter { .. }
    ));
    assert_eq!(model.cluster_centers(), snapshot_centers.view());
    assert_eq!(model.counts(), snapshot_counts.view());
    assert_eq!(model.batches_processed(), 1);
}

#[test]
fn minibatch_validation_rejects_non_finite_values_and_feature_mismatch() {
    let estimator = SKMiniBatchKMeans::<f64>::build()
        .number_of_clusters(2)
        .build()
        .unwrap();

    let dense_nan = array![[0.0_f64, 1.0], [2.0, f64::NAN], [4.0, 5.0]];
    assert!(matches!(
        estimator.fit(&dense_nan).unwrap_err(),
        SKError::NonFiniteValue { position: 3 }
    ));

    let sparse_inf = csr(&[(0, 0, 1.0), (2, 1, f64::NEG_INFINITY)], (3, 3));
    assert!(matches!(
        estimator.fit(sparse_inf.view()).unwrap_err(),
        SKError::NonFiniteValue { position: 7 }
    ));

    let mut model = estimator
        .fit(&array![[0.0_f64, 0.0], [0.1, 0.2], [10.0, 10.0]])
        .unwrap();
    assert!(matches!(
        model.partial_fit(&dense_nan).unwrap_err(),
        SKError::NonFiniteValue { position: 3 }
    ));
    assert!(matches!(
        model.predict(&dense_nan).unwrap_err(),
        SKError::NonFiniteValue { position: 3 }
    ));
    assert!(matches!(
        model.transform(&dense_nan).unwrap_err(),
        SKError::NonFiniteValue { position: 3 }
    ));

    let wrong_width = array![[0.0_f64, 0.0, 0.0], [1.0, 1.0, 1.0]];
    assert!(matches!(
        model.partial_fit(&wrong_width).unwrap_err(),
        SKError::ShapeMismatch { .. }
    ));
    assert!(matches!(
        model.predict(&wrong_width).unwrap_err(),
        SKError::ShapeMismatch { .. }
    ));
    assert!(matches!(
        model.transform(&wrong_width).unwrap_err(),
        SKError::ShapeMismatch { .. }
    ));

    let mut mismatched_stream = VecSource(VecDeque::from([
        SKDataBatch::new_dense(array![[0.0_f64, 0.0], [10.0, 10.0]], 0, false),
        SKDataBatch::new_dense(wrong_width, 1, true),
    ]));
    assert!(matches!(
        estimator.fit_streaming(&mut mismatched_stream).unwrap_err(),
        SKError::ShapeMismatch { .. }
    ));
}
