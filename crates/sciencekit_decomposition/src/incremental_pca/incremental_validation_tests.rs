//! Companion validation tests for empty inputs, component-count boundaries and
//! batch-size restrictions on `SKIncrementalPCA` (spec `incremental-pca`).

use std::collections::VecDeque;

use ndarray::{Array2, array};
use sciencekit_common::{SKBuilder, SKDataBatch, SKError, SKLazySource, SKUnsupervisedFit};

use super::SKIncrementalPCA;

pub(super) struct SliceBatchSource {
    pub(super) batches: VecDeque<SKDataBatch<f64>>,
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
