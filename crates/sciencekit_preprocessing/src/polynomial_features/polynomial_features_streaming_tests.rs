//! Streaming tests for `SKPolynomialFeatures` (spec `polynomial-features`).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array2, array};
use sciencekit_common::{SKDataBatch, SKError, SKLazySource, SKReopenableSource};

use super::model::SKPolynomialFeaturesModel;
use super::polynomial_features_tests::default_expander;

/// A small in-memory source for the streaming tests.
struct VecSource {
    batches: Vec<Array2<f64>>,
    next: usize,
}

impl VecSource {
    fn new(batches: Vec<Array2<f64>>) -> Self {
        Self { batches, next: 0 }
    }
}

impl SKLazySource<f64> for VecSource {
    type Error = SKError;

    fn next_batch(&mut self) -> Result<Option<SKDataBatch<f64>>, Self::Error> {
        if self.next >= self.batches.len() {
            return Ok(None);
        }
        let position = self.next;
        self.next += 1;
        let is_final = self.next == self.batches.len();
        Ok(Some(SKDataBatch::new(
            self.batches[position].clone(),
            position,
            is_final,
        )))
    }
}

/// Uneven streaming batches learn the same expansion as the dense fit.
#[test]
fn streaming_uneven_batches_equal_dense() {
    let all = Array2::from_shape_vec((3, 2), vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let reference = default_expander().fit(all.view()).unwrap();

    let mut source = VecSource::new(vec![
        array![[1.0_f64, 2.0], [3.0, 4.0]],
        array![[5.0_f64, 6.0]],
    ]);
    let streamed: SKPolynomialFeaturesModel =
        default_expander().fit_streaming(&mut source).unwrap();

    assert_eq!(
        streamed.number_of_input_features(),
        reference.number_of_input_features()
    );
    let expected: Array2<f64> = reference.transform_dense(all.view()).unwrap();
    let actual: Array2<f64> = streamed.transform_dense(all.view()).unwrap();
    assert_eq!(actual, expected);
}

/// A reopenable provider takes exactly one pass and learns the dense expansion.
#[test]
fn streaming_reopenable_provider_takes_one_pass() {
    let all = Array2::from_shape_vec((3, 2), vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
    let reference = default_expander().fit(all.view()).unwrap();
    let batches = vec![array![[1.0_f64, 2.0], [3.0, 4.0]], array![[5.0_f64, 6.0]]];
    let factory_calls = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&factory_calls);
    let provider = SKReopenableSource::new(move || {
        counted.fetch_add(1, Ordering::SeqCst);
        Ok(VecSource::new(batches.clone()))
    });
    let streamed = default_expander().fit_streaming(provider).unwrap();
    assert_eq!(factory_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        streamed.number_of_input_features(),
        reference.number_of_input_features()
    );
    let expected: Array2<f64> = reference.transform_dense(all.view()).unwrap();
    let actual: Array2<f64> = streamed.transform_dense(all.view()).unwrap();
    assert_eq!(actual, expected);
}

/// An empty streaming source fails instead of producing a vacuous model.
#[test]
fn streaming_empty_source_errors() {
    let mut source = VecSource::new(Vec::new());
    let result = default_expander().fit_streaming(&mut source);
    assert!(result.is_err());
}

/// Streaming rejects zero-feature input instead of building a broken model.
#[test]
fn streaming_zero_features_errors() {
    let mut source = VecSource::new(vec![Array2::<f64>::zeros((3, 0))]);
    let result = default_expander().fit_streaming(&mut source);
    assert!(matches!(result, Err(SKError::ShapeMismatch { .. })));
}
