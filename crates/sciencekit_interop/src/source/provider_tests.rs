//! Tests for the tabular source provider bridge (spec
//! `dataset-interop-core`).

use sciencekit_common::{
    SKError, SKLazySource, SKSourceProvider, SKSupervisedLazySource, SKSupervisedSourceProvider,
};

use super::{SKInMemoryTabularSource, SKTabularSource, SKTabularSourceProvider};
use crate::batch::SKFeatureLayout;
use crate::missing::SKMissingValuePolicy;
use crate::selection::SKColumnSelection;
use crate::source::test_fixtures::{fixture_batch, fixture_batches};

fn feature_selection() -> SKColumnSelection {
    SKColumnSelection::builder()
        .feature("x")
        .feature("y")
        .build()
        .unwrap()
}

fn supervised_selection() -> SKColumnSelection {
    SKColumnSelection::builder()
        .feature("y")
        .target("x")
        .build()
        .unwrap()
}

fn feature_provider_is_reopenable<P: SKSourceProvider<f64>>(provider: &P) -> bool {
    P::IS_REOPENABLE
}

fn provider<Source: SKTabularSource>(
    source: Source,
    selection: SKColumnSelection,
) -> SKTabularSourceProvider<Source, f64> {
    SKTabularSourceProvider::new(
        source,
        selection,
        SKMissingValuePolicy::Reject,
        SKFeatureLayout::Dense,
    )
}

/// A reopenable source feeds several passes from the beginning each time.
#[test]
fn reopenable_source_feeds_multiple_passes() {
    let mut provider = provider(
        SKInMemoryTabularSource::new(fixture_batches(3)),
        feature_selection(),
    );
    assert!(feature_provider_is_reopenable(&provider));

    for _pass in 0..2 {
        let mut source = SKSourceProvider::<f64>::open(&mut provider).unwrap();
        let mut positions = Vec::new();
        while let Some(batch) = source.next_batch().unwrap() {
            positions.push(batch.position());
        }
        assert_eq!(positions, vec![0, 1, 2]);
    }
}

/// A one-shot source feeds one pass and rejects a second open.
#[test]
fn one_shot_source_feeds_one_pass() {
    #[derive(Debug)]
    struct OneShot {
        batches: Vec<crate::batch::SKTabularBatch>,
        cursor: usize,
    }

    impl SKTabularSource for OneShot {
        fn open(&mut self) -> Result<(), SKError> {
            self.cursor = 0;
            Ok(())
        }

        fn next_batch(&mut self) -> Result<Option<crate::batch::SKTabularBatch>, SKError> {
            let batch = self.batches.get(self.cursor).cloned();
            if batch.is_some() {
                self.cursor += 1;
            }
            Ok(batch)
        }
    }

    let mut provider = provider(
        OneShot {
            batches: fixture_batches(2),
            cursor: 0,
        },
        feature_selection(),
    );
    assert!(!feature_provider_is_reopenable(&provider));

    {
        let mut source = SKSourceProvider::<f64>::open(&mut provider).unwrap();
        let mut count = 0;
        while source.next_batch().unwrap().is_some() {
            count += 1;
        }
        assert_eq!(count, 2);
    }
    let error = SKSourceProvider::<f64>::open(&mut provider).unwrap_err();
    assert!(matches!(error, SKError::UnsupportedOperation { .. }));
}

/// The supervised provider pairs features with the selected target.
#[test]
fn supervised_provider_converts_targets() {
    let mut provider = provider(
        SKInMemoryTabularSource::new(fixture_batches(2)),
        supervised_selection(),
    );
    let mut source = SKSupervisedSourceProvider::<f64>::open(&mut provider).unwrap();
    let first = source.next_supervised_batch().unwrap().unwrap();
    assert_eq!(first.position(), 0);
    assert_eq!(first.targets().to_vec(), vec![0.0]);
    let second = source.next_supervised_batch().unwrap().unwrap();
    assert_eq!(second.targets().to_vec(), vec![1.0]);
    assert!(source.next_supervised_batch().unwrap().is_none());
}

/// A supervised provider requires a declared target.
#[test]
fn supervised_provider_requires_a_target() {
    let mut provider = provider(
        SKInMemoryTabularSource::new(fixture_batches(1)),
        feature_selection(),
    );
    let error = SKSupervisedSourceProvider::<f64>::open(&mut provider).unwrap_err();
    assert!(matches!(error, SKError::UnsupportedOperation { .. }));
}

/// A read failure surfaces through the provider pass.
#[test]
fn read_failure_surfaces_through_the_pass() {
    #[derive(Debug)]
    struct Failing {
        opened: bool,
        delivered: bool,
    }

    impl SKTabularSource for Failing {
        const IS_REOPENABLE: bool = true;

        fn open(&mut self) -> Result<(), SKError> {
            self.opened = true;
            self.delivered = false;
            Ok(())
        }

        fn next_batch(&mut self) -> Result<Option<crate::batch::SKTabularBatch>, SKError> {
            if !self.delivered {
                self.delivered = true;
                return Ok(Some(fixture_batch(0, false)));
            }
            Err(SKError::Conversion("engine read failed".to_string()))
        }
    }

    let mut provider = provider(
        Failing {
            opened: false,
            delivered: false,
        },
        feature_selection(),
    );
    let mut source = SKSourceProvider::<f64>::open(&mut provider).unwrap();
    assert!(source.next_batch().unwrap().is_some());
    let error = source.next_batch().unwrap_err();
    assert!(error.to_string().contains("engine read failed"));
}
