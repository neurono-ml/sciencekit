//! Streaming companion tests for the min/max scaler: dense-equivalence across
//! batchings, streaming error paths and the large-scale run. Shared datasets
//! and helpers live in `test_datasets`.

use ndarray::array;
use sciencekit_common::{
    SKBuilder, SKError, SKExecutionMode, SKFeatureTransformer, SKUnsupervisedFit,
};

use super::test_datasets::{
    VecSource, assert_close, chunked_source, default_scaler, large_features, small_features,
};
use crate::minmax_scaler::core_implementation::SKMinMaxScaler;

/// Streaming over uneven batches learns the same model as the dense fit,
/// exactly.
#[test]
fn streaming_matches_dense_fit() {
    let features = small_features();
    let reference = default_scaler().fit(&features).expect("fit succeeds");
    let mut source = chunked_source(&features, &[2, 1, 3]);
    let streamed = default_scaler()
        .fit_streaming(&mut source)
        .expect("streaming fit succeeds");
    assert_eq!(streamed.minimum(), reference.minimum());
    assert_eq!(streamed.range(), reference.range());
    let output = streamed.transform(&features).expect("transform succeeds");
    let expected = reference.transform(&features).expect("transform succeeds");
    assert_eq!(output, expected);
}

/// A single-batch stream degenerates to the dense result.
#[test]
fn streaming_single_batch_matches_dense_fit() {
    let features = small_features();
    let reference = default_scaler().fit(&features).expect("fit succeeds");
    let mut source = chunked_source(&features, &[6]);
    let streamed = default_scaler()
        .fit_streaming(&mut source)
        .expect("streaming fit succeeds");
    assert_eq!(streamed.minimum(), reference.minimum());
}

/// An empty source fails instead of producing a vacuous model.
#[test]
fn streaming_empty_source_errors() {
    let mut source = chunked_source(&ndarray::Array2::<f64>::zeros((0, 2)), &[]);
    let error = default_scaler()
        .fit_streaming(&mut source)
        .expect_err("empty stream must fail");
    assert!(matches!(error, SKError::Conversion(_)));
}

/// Batches disagreeing on column count fail with a shape error.
#[test]
fn streaming_column_mismatch_errors() {
    let mut source = VecSource::new(vec![
        array![[1.0, 2.0], [3.0, 4.0]],
        array![[5.0, 6.0, 7.0]],
    ]);
    let error = default_scaler()
        .fit_streaming(&mut source)
        .expect_err("mismatched stream must fail");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// The builder flags propagate through the streaming path.
#[test]
fn streaming_respects_builder_flags() {
    let features = small_features();
    let scaler = SKMinMaxScaler::<f64>::build()
        .feature_range((-1.0, 1.0))
        .clip(true)
        .build()
        .expect("scaler builds");
    let reference = scaler.fit(&features).expect("fit succeeds");
    let mut source = chunked_source(&features, &[4, 2]);
    let streamed = scaler
        .fit_streaming(&mut source)
        .expect("streaming fit succeeds");
    assert_close(streamed.minimum(), &reference.minimum().to_vec());
    assert_close(streamed.range(), &reference.range().to_vec());
    assert_eq!(streamed.feature_range(), (-1.0, 1.0));
    assert!(streamed.clip());
}

/// An explicit in-memory intent collects the batches through the dense path.
#[test]
fn streaming_explicit_in_process_collects_batches() {
    let features = small_features();
    let reference = default_scaler().fit(&features).expect("fit succeeds");
    let scaler = SKMinMaxScaler::<f64>::build()
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .expect("scaler builds");
    let mut source = chunked_source(&features, &[2, 4]);
    let collected = scaler
        .fit_streaming(&mut source)
        .expect("streaming fit succeeds");
    assert_eq!(collected.minimum(), reference.minimum());
    assert_eq!(collected.range(), reference.range());
}

/// Streaming stays generic over both supported floats.
#[test]
fn streaming_accepts_either_float() {
    let batch_f32 = array![[1.0_f32, 2.0], [3.0, 4.0], [5.0, 6.0]];
    let mut source = chunked_source(&batch_f32, &[1, 2]);
    let streamed = SKMinMaxScaler::<f32>::build()
        .build()
        .expect("scaler builds")
        .fit_streaming(&mut source)
        .expect("streaming fit succeeds");
    assert_eq!(streamed.minimum().clone(), array![1.0_f32, 2.0]);
}

/// Streaming the large dataset in uneven batches matches the dense fit exactly.
#[test]
fn large_streaming_matches_dense_fit() {
    let features = large_features();
    let reference = default_scaler().fit(&features).expect("fit succeeds");
    let mut source = chunked_source(&features, &[40_000, 17_500, 42_500]);
    let streamed = default_scaler()
        .fit_streaming(&mut source)
        .expect("streaming fit succeeds");
    assert_eq!(streamed.minimum(), reference.minimum());
    assert_eq!(streamed.range(), reference.range());
}
