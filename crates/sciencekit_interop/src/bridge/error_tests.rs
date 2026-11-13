//! Tests for the bridge error path (spec `dataset-interop-core`).

use sciencekit_common::SKError;

use super::SKAsyncTabularSource;
use crate::bridge::test_streams::{ScriptedStream, StreamProbe};
use crate::source::SKTabularSource;

/// An asynchronous stream error surfaces after the batches already delivered,
/// and the pass then ends cleanly.
#[test]
fn stream_error_surfaces_after_delivered_batches() {
    let probe = StreamProbe::default();
    let mut source = SKAsyncTabularSource::new(ScriptedStream::failing(10, 2, probe), 1);
    source.open().unwrap();

    assert_eq!(source.next_batch().unwrap().unwrap().position(), 0);
    assert_eq!(source.next_batch().unwrap().unwrap().position(), 1);

    let error = source.next_batch().unwrap_err();
    match &error {
        SKError::Conversion(message) => assert!(message.contains("engine stream failed")),
        other => panic!("expected a conversion error, got {other:?}"),
    }

    assert!(
        source.next_batch().unwrap().is_none(),
        "the pass ends after the error"
    );
}
