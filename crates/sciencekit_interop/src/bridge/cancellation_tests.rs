//! Tests for bridge cancellation (spec `dataset-interop-core`).

use super::SKAsyncTabularSource;
use crate::bridge::test_streams::{ScriptedStream, StreamProbe};
use crate::source::SKTabularSource;

/// Dropping the consumer closes the channel, cancels the reader and drops the
/// stream, without panicking or leaking the thread.
#[test]
fn dropping_the_source_cancels_and_drops_the_stream() {
    let probe = StreamProbe::default();
    let mut source = SKAsyncTabularSource::new(ScriptedStream::new(100, probe.clone()), 1);
    source.open().unwrap();

    assert!(source.next_batch().unwrap().is_some());
    assert!(
        !probe.was_dropped(),
        "the stream is still owned by the reader before the drop"
    );

    drop(source);
    assert!(
        probe.was_dropped(),
        "dropping the source must drop the stream and join the reader thread"
    );
}

/// A one-shot bridge accepts a single open and rejects reopening.
#[test]
fn bridge_is_one_shot() {
    let probe = StreamProbe::default();
    let mut source = SKAsyncTabularSource::new(ScriptedStream::new(1, probe), 1);
    source.open().unwrap();
    let error = source.open().unwrap_err();
    assert!(matches!(
        error,
        sciencekit_common::SKError::UnsupportedOperation { .. }
    ));
}
