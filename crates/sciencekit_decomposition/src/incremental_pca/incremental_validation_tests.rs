//! Companion validation tests for `SKIncrementalPCA` and
//! `SKIncrementalPCAModel` (spec `incremental-pca`).

use std::collections::VecDeque;

use ndarray::{Array2, array};
use sciencekit_common::{SKBuilder, SKDataBatch, SKError, SKLazySource, SKUnsupervisedFit};
use sprs::TriMat;

use super::SKIncrementalPCA;

struct SliceBatchSource {
    batches: VecDeque<SKDataBatch<f64>>,
}

impl SKLazySource<f64> for SliceBatchSource {
    type Error = SKError;

    fn next_batch(&mut self) -> Result<Option<SKDataBatch<f64>>, Self::Error> {
        Ok(self.batches.pop_front())
    }
}

#[test]
fn incremental_validation_rejects_empty_inputs_everywhere() {
    let estimator = SKIncrementalPCA::<f64>::build().build().unwrap();
    assert!(matches!(
        estimator.fit(&Array2::<f64>::zeros((0, 4))).unwrap_err(),
        SKError::EmptyInput { operation: "fit" }
    ));
    assert!(matches!(
        estimator.fit(&Array2::<f64>::zeros((4, 0))).unwrap_err(),
        SKError::EmptyInput { operation: "fit" }
    ));
    assert!(matches!(
        estimator
            .fit_transform(&Array2::<f64>::zeros((0, 4)))
            .unwrap_err(),
        SKError::EmptyInput {
            operation: "fit_transform"
        }
    ));

    let mut empty_stream = SliceBatchSource {
        batches: VecDeque::new(),
    };
    assert!(matches!(
        estimator.fit_streaming(&mut empty_stream).unwrap_err(),
        SKError::EmptyInput {
            operation: "fit_streaming"
        }
    ));

    let mut mid_empty_stream = SliceBatchSource {
        batches: VecDeque::from(vec![
            SKDataBatch::new_dense(Array2::ones((4, 3)), 0, false),
            SKDataBatch::new_dense(Array2::zeros((0, 3)), 1, true),
        ]),
    };
    assert!(matches!(
        estimator.fit_streaming(&mut mid_empty_stream).unwrap_err(),
        SKError::EmptyInput {
            operation: "fit_streaming"
        }
    ));

    let initial = array![
        [1.0_f64, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [7.0, 8.0, 9.0],
        [10.0, 11.0, 12.0]
    ];
    let mut model = SKIncrementalPCA::<f64>::build()
        .number_of_components(2)
        .build()
        .unwrap()
        .fit(&initial)
        .unwrap();
    assert!(matches!(
        model
            .partial_fit(&Array2::<f64>::zeros((0, 3)))
            .unwrap_err(),
        SKError::EmptyInput {
            operation: "partial_fit"
        }
    ));
    assert!(matches!(
        model.transform(&Array2::<f64>::zeros((0, 3))).unwrap_err(),
        SKError::EmptyInput {
            operation: "transform"
        }
    ));
    assert!(matches!(
        model
            .inverse_transform(&Array2::<f64>::zeros((0, 2)))
            .unwrap_err(),
        SKError::EmptyInput {
            operation: "inverse_transform"
        }
    ));
}

#[test]
fn incremental_validation_rejects_excessive_components_and_small_batches() {
    assert!(matches!(
        SKIncrementalPCA::<f64>::build()
            .number_of_components(4)
            .batch_size(2)
            .build()
            .unwrap_err(),
        SKError::InvalidHyperparameter {
            name: "batch_size",
            ..
        }
    ));

    let small_rows = array![
        [1.0_f64, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]
    ];
    assert!(matches!(
        SKIncrementalPCA::<f64>::build()
            .number_of_components(4)
            .build()
            .unwrap()
            .fit(&small_rows)
            .unwrap_err(),
        SKError::InvalidHyperparameter {
            name: "number_of_components",
            ..
        }
    ));

    let wider = Array2::from_shape_fn((10, 4), |(row, column)| (row * 3 + column) as f64);
    assert!(matches!(
        SKIncrementalPCA::<f64>::build()
            .batch_size(2)
            .build()
            .unwrap()
            .fit(&wider)
            .unwrap_err(),
        SKError::InvalidHyperparameter {
            name: "batch_size",
            ..
        }
    ));

    let mut undersized_stream = SliceBatchSource {
        batches: VecDeque::from(vec![SKDataBatch::new_dense(wider.clone(), 0, true)]),
    };
    assert!(matches!(
        SKIncrementalPCA::<f64>::build()
            .batch_size(2)
            .build()
            .unwrap()
            .fit_streaming(&mut undersized_stream)
            .unwrap_err(),
        SKError::InvalidHyperparameter {
            name: "batch_size",
            ..
        }
    ));

    let mut model = SKIncrementalPCA::<f64>::build()
        .number_of_components(3)
        .build()
        .unwrap()
        .fit(&wider)
        .unwrap();
    let before_count = model.samples_seen();
    let before_mean = model.mean().to_owned();

    let two_rows = array![[1.0_f64, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
    assert!(matches!(
        model.partial_fit(&two_rows).unwrap_err(),
        SKError::InvalidHyperparameter {
            name: "batch_size",
            ..
        }
    ));
    assert_eq!(model.samples_seen(), before_count);
    assert_eq!(model.mean(), before_mean.view());
}

#[test]
fn incremental_validation_rejects_feature_mismatch_non_finite_and_csc() {
    let valid = Array2::from_shape_fn((8, 4), |(row, column)| {
        ((row + 1) * (column + 2)) as f64 + (row % 3) as f64 * 0.5
    });
    let mut model = SKIncrementalPCA::<f64>::build()
        .number_of_components(3)
        .build()
        .unwrap()
        .fit(&valid)
        .unwrap();
    let prior_samples = model.samples_seen();

    let wrong_columns = Array2::<f64>::ones((4, 5));
    assert!(matches!(
        model.partial_fit(&wrong_columns).unwrap_err(),
        SKError::ShapeMismatch { expected, found } if expected == vec![4, 4] && found == vec![4, 5]
    ));
    assert_eq!(model.samples_seen(), prior_samples);

    let mut mismatched_stream = SliceBatchSource {
        batches: VecDeque::from(vec![
            SKDataBatch::new_dense(valid.clone(), 0, false),
            SKDataBatch::new_dense(wrong_columns, 1, true),
        ]),
    };
    assert!(matches!(
        SKIncrementalPCA::<f64>::build()
            .number_of_components(2)
            .build()
            .unwrap()
            .fit_streaming(&mut mismatched_stream)
            .unwrap_err(),
        SKError::ShapeMismatch { .. }
    ));

    let mut dense_nan = valid.clone();
    dense_nan[(1, 2)] = f64::NAN;
    assert!(matches!(
        SKIncrementalPCA::<f64>::build()
            .number_of_components(2)
            .build()
            .unwrap()
            .fit(&dense_nan)
            .unwrap_err(),
        SKError::NonFiniteValue { position: 6 }
    ));
    assert!(matches!(
        model.partial_fit(&dense_nan).unwrap_err(),
        SKError::NonFiniteValue { position: 6 }
    ));
    assert!(matches!(
        model.transform(&dense_nan).unwrap_err(),
        SKError::NonFiniteValue { position: 6 }
    ));

    let mut sparse_triplet = TriMat::new((4, 4));
    sparse_triplet.add_triplet(0, 1, 2.0_f64);
    sparse_triplet.add_triplet(2, 3, f64::INFINITY);
    let sparse_inf = sparse_triplet.to_csr();
    assert!(matches!(
        model.partial_fit(sparse_inf.view()).unwrap_err(),
        SKError::NonFiniteValue { position: 11 }
    ));

    let csc = {
        let mut triplet = TriMat::new((4, 4));
        triplet.add_triplet(0, 0, 1.0_f64);
        triplet.to_csc()
    };
    assert!(matches!(
        model.partial_fit(csc.view()).unwrap_err(),
        SKError::UnsupportedRepresentation {
            representation: "csc",
            ..
        }
    ));

    assert!(matches!(
        model.transform(&Array2::<f64>::zeros((2, 5))).unwrap_err(),
        SKError::ShapeMismatch { expected, found } if expected == vec![2, 4] && found == vec![2, 5]
    ));
    assert!(matches!(
        model
            .inverse_transform(&Array2::<f64>::zeros((2, 2)))
            .unwrap_err(),
        SKError::ShapeMismatch { expected, found } if expected == vec![2, 3] && found == vec![2, 2]
    ));
}
