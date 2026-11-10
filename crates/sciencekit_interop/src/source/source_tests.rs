//! Tests for the reopenable tabular source contract (spec
//! `dataset-interop-core`).

use sciencekit_common::SKError;

use super::{SKInMemoryTabularSource, SKTabularSource};
use crate::batch::SKTabularBatch;
use crate::source::test_fixtures::{fixture_batch, fixture_batches};

/// A reopenable source replays every batch from the beginning on each pass.
#[test]
fn reopenable_source_replays_each_pass() {
    let mut source = SKInMemoryTabularSource::new(fixture_batches(3));
    assert!(SKInMemoryTabularSource::IS_REOPENABLE);
    assert_eq!(source.len(), 3);

    for _pass in 0..2 {
        source.open().unwrap();
        let mut positions = Vec::new();
        while let Some(batch) = source.next_batch().unwrap() {
            positions.push(batch.position());
        }
        assert_eq!(positions, vec![0, 1, 2]);
    }
}

/// A one-shot source accepts one open and rejects reopening.
#[test]
fn one_shot_source_rejects_reopening() {
    struct OneShotSource {
        batch: Option<SKTabularBatch>,
        opened: bool,
    }

    impl SKTabularSource for OneShotSource {
        fn open(&mut self) -> Result<(), SKError> {
            if self.opened {
                return Err(SKError::UnsupportedOperation {
                    operation: "open_tabular_source",
                    reason: "the source is one-shot and was already opened".to_string(),
                });
            }
            self.opened = true;
            Ok(())
        }

        fn next_batch(&mut self) -> Result<Option<SKTabularBatch>, SKError> {
            Ok(self.batch.take())
        }
    }

    let mut source = OneShotSource {
        batch: Some(fixture_batch(0, true)),
        opened: false,
    };
    assert!(!OneShotSource::IS_REOPENABLE);
    source.open().unwrap();
    assert!(source.next_batch().unwrap().is_some());
    assert!(source.next_batch().unwrap().is_none());
    source.open().unwrap_err();
}

/// A read failure propagates as the central taxonomy error.
#[test]
fn read_failure_propagates_as_the_central_error() {
    struct FailingSource {
        batches: Vec<SKTabularBatch>,
        cursor: usize,
    }

    impl SKTabularSource for FailingSource {
        const IS_REOPENABLE: bool = true;

        fn open(&mut self) -> Result<(), SKError> {
            self.cursor = 0;
            Ok(())
        }

        fn next_batch(&mut self) -> Result<Option<SKTabularBatch>, SKError> {
            match self.cursor {
                0 => {
                    self.cursor += 1;
                    Ok(Some(fixture_batch(0, false)))
                }
                _ => Err(SKError::Conversion(
                    "the engine failed while reading the second batch".to_string(),
                )),
            }
        }
    }

    let mut source = FailingSource {
        batches: Vec::new(),
        cursor: 0,
    };
    source.open().unwrap();
    assert!(source.next_batch().unwrap().is_some());
    let error = source.next_batch().unwrap_err();
    match error {
        SKError::Conversion(message) => assert!(message.contains("engine failed")),
        other => panic!("expected a conversion error, got {other:?}"),
    }
}
