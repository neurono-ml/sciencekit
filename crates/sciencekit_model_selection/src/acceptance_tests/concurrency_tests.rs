//! Acceptance: splitters are shareable across threads and concurrent runs agree.

use ndarray::Array2;
use sciencekit_common::{SKBuilder, SKDataView};

use super::fixtures::dense_design;
use crate::SKSplitter;
use crate::split::group::{SKGroupKFold, SKGroupKFoldBuilder};
use crate::split::kfold::{SKKFold, SKKFoldBuilder};
use crate::split::stratified::{SKStratifiedKFold, SKStratifiedKFoldBuilder};
use crate::train_test_split::{SKTrainTestSplitBuilder, SKTrainTestSplitter};

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn acceptance_concurrency_every_splitter_and_builder_is_send_and_sync() {
    assert_send_sync::<SKKFold>();
    assert_send_sync::<SKKFoldBuilder>();
    assert_send_sync::<SKStratifiedKFold>();
    assert_send_sync::<SKStratifiedKFoldBuilder>();
    assert_send_sync::<SKGroupKFold>();
    assert_send_sync::<SKGroupKFoldBuilder>();
    assert_send_sync::<SKTrainTestSplitter>();
    assert_send_sync::<SKTrainTestSplitBuilder>();
    assert_send_sync::<crate::SKIndexSplit>();
}

#[test]
fn acceptance_concurrency_shared_splitter_generates_identical_folds_per_thread() {
    let features: Array2<f64> = dense_design(500, 6);
    let splitter = SKKFold::build()
        .number_of_splits(5)
        .shuffle(true)
        .random_seed(42)
        .build()
        .unwrap();
    let expected = splitter
        .split(SKDataView::Dense(features.view()), None, None)
        .unwrap();

    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..4 {
            let splitter = &splitter;
            let features = &features;
            handles.push(scope.spawn(move || {
                splitter
                    .split(SKDataView::Dense(features.view()), None, None)
                    .unwrap()
            }));
        }
        for handle in handles {
            assert_eq!(handle.join().expect("the thread joins"), expected);
        }
    });
}

#[test]
fn acceptance_concurrency_shared_train_test_splitter_is_deterministic() {
    let features: Array2<f64> = dense_design(300, 3);
    let splitter = SKTrainTestSplitter::build()
        .test_fraction(0.3)
        .random_seed(7)
        .build()
        .unwrap();
    let expected = splitter
        .split(SKDataView::Dense(features.view()), None, None)
        .unwrap();

    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..3 {
            let splitter = &splitter;
            let features = &features;
            handles.push(scope.spawn(move || {
                splitter
                    .split(SKDataView::Dense(features.view()), None, None)
                    .unwrap()
            }));
        }
        for handle in handles {
            assert_eq!(handle.join().expect("the thread joins"), expected);
        }
    });
}
