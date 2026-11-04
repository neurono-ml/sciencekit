//! Companion tests for the reopenable provider, the error paths and the
//! backend carried by the out-of-core fit (spec `simple-imputer`).

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::array;
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKError, SKExecutionMode, SKReopenableSource, SKUnsupervisedFit,
};

use super::core_implementation::SKSimpleImputer;
use super::streaming_tests::{SKVectorSource, chunk_rows};

#[test]
fn reopenable_provider_takes_one_pass_and_matches_the_dense_fit() {
    let features = array![[1.0, 10.0], [3.0, f64::NAN], [5.0, 14.0], [7.0, 18.0]];
    let reference = SKSimpleImputer::<f64>::build()
        .build()
        .unwrap()
        .fit(&features)
        .unwrap();
    let batches = chunk_rows(&features, &[1, 2, 1]);
    let factory_calls = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&factory_calls);
    let provider = SKReopenableSource::new(move || {
        counted.fetch_add(1, Ordering::SeqCst);
        Ok(SKVectorSource::new(batches.clone()))
    });
    let streamed = SKSimpleImputer::<f64>::build()
        .execution_mode(SKExecutionMode::OutOfCoreStreaming)
        .build()
        .unwrap()
        .fit_streaming(provider)
        .unwrap();
    assert_eq!(factory_calls.load(Ordering::SeqCst), 1);
    assert_eq!(streamed.statistics(), reference.statistics());
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
