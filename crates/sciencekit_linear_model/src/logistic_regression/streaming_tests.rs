//! Companion tests for the out-of-core logistic fit (spec
//! `logistic-regression`).

use ndarray::{Array1, Array2, array};
use sciencekit_common::builders::SKBuilder;
use sciencekit_common::{
    SKClassifierPredictor, SKError, SKSupervisedBatch, SKSupervisedFit, SKSupervisedLazySource,
};

use super::SKLogisticRegression;

/// An in-memory supervised source: each entry is one owned batch.
struct VecSource<F> {
    batches: Vec<(Array2<F>, Array1<F>)>,
    next: usize,
}

impl<F: sciencekit_common::SKFloat> SKSupervisedLazySource<F> for VecSource<F> {
    type Error = SKError;

    fn next_supervised_batch(&mut self) -> Result<Option<SKSupervisedBatch<F>>, Self::Error> {
        if self.next >= self.batches.len() {
            return Ok(None);
        }
        let position = self.next;
        self.next += 1;
        let (features, targets) = self.batches[position].clone();
        let is_final = self.next == self.batches.len();
        Ok(Some(SKSupervisedBatch::new(
            features, targets, position, is_final,
        )))
    }
}

fn batches() -> Vec<(Array2<f64>, Array1<f64>)> {
    vec![
        (
            array![[-2.0, -1.0], [-1.0, -2.0], [-1.0, -1.0]],
            array![0.0, 0.0, 0.0],
        ),
        (
            array![[1.0, 1.0], [2.0, 1.0], [1.0, 2.0]],
            array![1.0, 1.0, 1.0],
        ),
    ]
}

#[test]
fn streaming_equals_the_dense_fit() {
    let all_features = array![
        [-2.0, -1.0],
        [-1.0, -2.0],
        [-1.0, -1.0],
        [1.0, 1.0],
        [2.0, 1.0],
        [1.0, 2.0]
    ];
    let all_labels = array![0_i64, 0, 0, 1, 1, 1];
    let dense = SKLogisticRegression::<f64>::build()
        .build()
        .unwrap()
        .fit(all_features.view(), all_labels.view())
        .unwrap();
    let mut source = VecSource {
        batches: batches(),
        next: 0,
    };
    let streamed = SKLogisticRegression::<f64>::build()
        .build()
        .unwrap()
        .fit_streaming(&mut source)
        .unwrap();
    for (dense_value, streamed_value) in dense.coefficients().iter().zip(streamed.coefficients()) {
        assert!((dense_value - streamed_value).abs() < 1e-6);
    }
    let predicted = streamed.predict_labels(all_features.view()).unwrap();
    assert_eq!(predicted.to_vec(), all_labels.to_vec());
}

#[test]
fn an_empty_source_is_rejected() {
    let mut source = VecSource::<f64> {
        batches: Vec::new(),
        next: 0,
    };
    let result = SKLogisticRegression::<f64>::build()
        .build()
        .unwrap()
        .fit_streaming(&mut source);
    assert!(result.is_err());
}
