//! Companion tests for the out-of-core fit (spec `simple-imputer`).

use ndarray::{Array2, array, s};
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKDataBatch, SKError, SKExecutionMode, SKFeatureTransformer, SKFloat,
    SKLazySource, SKUnsupervisedFit,
};

use super::core_implementation::SKSimpleImputer;
use super::strategy::SKImputationStrategy;

/// The documented relative tolerance of the online median sketch.
const SKETCH_RELATIVE_TOLERANCE: f64 = 0.05;

/// A finite source replaying in-memory batches, mirroring the scaler fixture.
struct SKVectorSource<F> {
    batches: Vec<Array2<F>>,
    next: usize,
}

impl<F> SKVectorSource<F> {
    fn new(batches: Vec<Array2<F>>) -> Self {
        Self { batches, next: 0 }
    }
}

impl<F: SKFloat> SKLazySource<F> for SKVectorSource<F> {
    type Error = SKError;

    fn next_batch(&mut self) -> Result<Option<SKDataBatch<F>>, Self::Error> {
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

/// Split a matrix into row chunks of the given lengths.
fn chunk_rows<F: SKFloat>(features: &Array2<F>, lengths: &[usize]) -> Vec<Array2<F>> {
    let mut batches = Vec::new();
    let mut start = 0;
    for &length in lengths {
        let end = (start + length).min(features.nrows());
        batches.push(features.slice(s![start..end, ..]).to_owned());
        start = end;
    }
    batches
}

/// A deterministic column of values with every seventh cell missing.
fn median_features(rows_count: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows_count, 1), |(row, _)| {
        if row % 7 == 0 {
            f64::NAN
        } else {
            ((row * 37) % rows_count) as f64 + 1.0
        }
    })
}

#[test]
fn in_process_streaming_matches_the_dense_fit_exactly() {
    let features = array![
        [1.0, 10.0],
        [3.0, f64::NAN],
        [5.0, 14.0],
        [7.0, 18.0],
        [9.0, f64::NAN]
    ];
    let reference = SKSimpleImputer::<f64>::build()
        .build()
        .unwrap()
        .fit(&features)
        .unwrap();
    let mut source = SKVectorSource::new(chunk_rows(&features, &[2, 3]));
    let streamed = SKSimpleImputer::<f64>::build()
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap()
        .fit_streaming(&mut source)
        .unwrap();
    assert_eq!(streamed.statistics(), reference.statistics());
    assert_eq!(
        streamed.transform(&array![[f64::NAN, f64::NAN]]).unwrap(),
        reference.transform(&array![[f64::NAN, f64::NAN]]).unwrap()
    );
}

#[test]
fn streaming_driver_mean_is_exact() {
    let features = array![[1.0, 10.0], [3.0, f64::NAN], [5.0, 14.0], [7.0, 18.0]];
    let reference = SKSimpleImputer::<f64>::build()
        .build()
        .unwrap()
        .fit(&features)
        .unwrap();
    let mut source = SKVectorSource::new(chunk_rows(&features, &[1, 2, 1]));
    let streamed = SKSimpleImputer::<f64>::build()
        .build()
        .unwrap()
        .fit_streaming(&mut source)
        .unwrap();
    assert_eq!(streamed.statistics(), reference.statistics());
}

#[test]
fn streaming_driver_mode_is_exact() {
    let features = array![[2.0], [5.0], [2.0], [5.0], [2.0], [8.0]];
    let reference = SKSimpleImputer::<f64>::build()
        .strategy(SKImputationStrategy::MostFrequent)
        .build()
        .unwrap()
        .fit(&features)
        .unwrap();
    let mut source = SKVectorSource::new(chunk_rows(&features, &[2, 1, 3]));
    let streamed = SKSimpleImputer::<f64>::build()
        .strategy(SKImputationStrategy::MostFrequent)
        .build()
        .unwrap()
        .fit_streaming(&mut source)
        .unwrap();
    assert_eq!(streamed.statistics(), reference.statistics());
}

#[test]
fn streaming_driver_median_stays_within_the_sketch_tolerance() {
    let features = median_features(600);
    let reference = SKSimpleImputer::<f64>::build()
        .strategy(SKImputationStrategy::Median)
        .build()
        .unwrap()
        .fit(&features)
        .unwrap();
    let mut source = SKVectorSource::new(chunk_rows(&features, &[121, 200, 279]));
    let streamed = SKSimpleImputer::<f64>::build()
        .strategy(SKImputationStrategy::Median)
        .build()
        .unwrap()
        .fit_streaming(&mut source)
        .unwrap();
    let expected = reference.statistics()[0];
    let actual = streamed.statistics()[0];
    let allowed = SKETCH_RELATIVE_TOLERANCE * expected.abs().max(1.0);
    assert!(
        (actual - expected).abs() <= allowed,
        "expected {expected}, found {actual} (tolerance {allowed})"
    );
}

#[test]
fn streaming_empty_source_errors() {
    let mut source = SKVectorSource::new(Vec::new());
    let error = SKSimpleImputer::<f64>::build()
        .build()
        .unwrap()
        .fit_streaming(&mut source)
        .err()
        .expect("an empty stream must fail");
    assert!(matches!(error, SKError::Conversion(_)));
}

#[test]
fn streaming_column_mismatch_errors() {
    let mut source = SKVectorSource::new(vec![
        array![[1.0, 2.0], [3.0, 4.0]],
        array![[5.0, 6.0, 7.0]],
    ]);
    let error = SKSimpleImputer::<f64>::build()
        .build()
        .unwrap()
        .fit_streaming(&mut source)
        .err()
        .expect("mismatched columns must fail");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

#[test]
fn streaming_carries_the_configured_backend() {
    let features = array![[1.0], [f64::NAN], [3.0]];
    let mut source = SKVectorSource::new(chunk_rows(&features, &[1, 2]));
    let streamed = SKSimpleImputer::<f64>::build()
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit_streaming(&mut source)
        .unwrap();
    assert_eq!(streamed.backend(), SKBackendKind::MatrixMultiply);
}
