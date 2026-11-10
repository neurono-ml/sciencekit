//! Tests for ordered delivery, bounded residency and thread separation (spec
//! `dataset-interop-core`).

use std::thread;

use super::SKAsyncTabularSource;
use crate::bridge::test_streams::{ScriptedStream, StreamProbe};
use crate::source::SKTabularSource;

/// Batches arrive in order and the resident count stays bounded by the depth.
#[test]
fn delivery_is_ordered_and_bounded() {
    let probe = StreamProbe::default();
    let mut source = SKAsyncTabularSource::new(ScriptedStream::new(6, probe.clone()), 2);
    source.open().unwrap();

    let mut positions = Vec::new();
    let mut received = 0usize;
    while let Some(batch) = source.next_batch().unwrap() {
        received += 1;
        assert!(
            probe.produced() <= received + 2 + 1,
            "at most depth + 1 batches may be resident: produced {}, received {}",
            probe.produced(),
            received
        );
        positions.push(batch.position());
    }
    assert_eq!(positions, vec![0, 1, 2, 3, 4, 5]);
}

/// The caller's thread never polls the asynchronous stream.
#[test]
fn caller_thread_never_polls() {
    let probe = StreamProbe::default();
    let mut source = SKAsyncTabularSource::new(ScriptedStream::new(3, probe.clone()), 1);
    source.open().unwrap();
    let caller = thread::current().id();
    let reader = source.reader_thread_identifier();

    while source.next_batch().unwrap().is_some() {}

    let poll_threads = probe.poll_threads();
    assert!(!poll_threads.is_empty());
    for polled_on in poll_threads {
        assert_ne!(polled_on, caller, "the caller thread must never poll");
        assert_eq!(polled_on, reader, "only the reader thread polls");
    }
}
