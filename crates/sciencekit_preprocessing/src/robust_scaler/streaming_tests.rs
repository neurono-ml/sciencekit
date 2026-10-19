//! Streaming companion tests for the robust scaler: the online sketch
//! approximating the exact dense fit, the in-memory collect branch and the
//! error paths. Shared datasets and helpers live in `test_datasets`.

use ndarray::array;
use sciencekit_common::{SKBuilder, SKError, SKExecutionMode, SKUnsupervisedFit};

use super::test_datasets::{
    SKETCH_RELATIVE_TOLERANCE, VecSource, assert_close_with_tolerance, chunked_source,
    default_scaler, large_features, small_features,
};
use crate::robust_scaler::core_implementation::SKRobustScaler;

/// Streaming over uneven batches approximates the exact dense fit within the
/// documented tolerance.
#[test]
fn streaming_approximates_dense_fit() {
    let features = large_features();
    let reference = default_scaler().fit(&features).expect("fit succeeds");
    let mut source = chunked_source(&features, &[40_000, 17_500, 42_500]);
    let streamed = default_scaler()
        .fit_streaming(&mut source)
        .expect("streaming fit succeeds");
    assert_eq!(streamed.sample_count(), 100_000);
    assert_close_with_tolerance(
        streamed.center(),
        &reference.center().to_vec(),
        SKETCH_RELATIVE_TOLERANCE,
    );
    assert_close_with_tolerance(
        streamed.scale(),
        &reference.scale().to_vec(),
        SKETCH_RELATIVE_TOLERANCE,
    );
}

/// An explicit in-memory intent collects the batches through the dense path, so
/// the result is exactly the dense fit.
#[test]
fn streaming_explicit_in_process_collects_batches() {
    let features = small_features();
    let reference = default_scaler().fit(&features).expect("fit succeeds");
    let scaler = SKRobustScaler::<f64>::build()
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .expect("scaler builds");
    let mut source = chunked_source(&features, &[2, 3]);
    let collected = scaler
        .fit_streaming(&mut source)
        .expect("streaming fit succeeds");
    assert_eq!(collected.center(), reference.center());
    assert_eq!(collected.scale(), reference.scale());
}

/// An empty source fails instead of producing a vacuous model.
#[test]
fn streaming_empty_source_errors() {
    let mut source = chunked_source(&ndarray::Array2::<f64>::zeros((0, 1)), &[]);
    let error = default_scaler()
        .fit_streaming(&mut source)
        .expect_err("empty stream must fail");
    assert!(matches!(error, SKError::Conversion(_)));
}

/// Batches disagreeing on column count fail with a shape error.
#[test]
fn streaming_column_mismatch_errors() {
    let mut source = VecSource::new(vec![array![[1.0, 2.0], [3.0, 4.0]], array![[5.0, 6.0, 7.0]]]);
    let error = default_scaler()
        .fit_streaming(&mut source)
        .expect_err("mismatched stream must fail");
    assert!(matches!(error, SKError::ShapeMismatch { .. }));
}

/// The builder flags and a target-range override survive the streaming path.
#[test]
fn streaming_respects_builder_flags() {
    let features = small_features();
    let scaler = SKRobustScaler::<f64>::build()
        .quantile_range((10.0, 90.0))
        .unit_variance(true)
        .build()
        .expect("scaler builds");
    let reference = scaler.fit(&features).expect("fit succeeds");
    let mut source = chunked_source(&features, &[2, 3]);
    let streamed = scaler
        .fit_streaming(&mut source)
        .expect("streaming fit succeeds");
    assert_eq!(streamed.quantile_range(), (10.0, 90.0));
    assert!(streamed.unit_variance());
    assert_close_with_tolerance(
        streamed.scale(),
        &reference.scale().to_vec(),
        SKETCH_RELATIVE_TOLERANCE,
    );
}

/// Streaming stays generic over both supported floats.
#[test]
fn streaming_accepts_either_float() {
    let batch_f32 = array![[1.0_f32], [2.0], [3.0], [4.0], [5.0]];
    let mut source = chunked_source(&batch_f32, &[2, 3]);
    let streamed = SKRobustScaler::<f32>::build()
        .build()
        .expect("scaler builds")
        .fit_streaming(&mut source)
        .expect("streaming fit succeeds");
    assert_eq!(streamed.center().len(), 1);
}
