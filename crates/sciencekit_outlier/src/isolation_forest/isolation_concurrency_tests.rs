//! Companion tests for repeated-fit determinism, thread-count invariance, concurrent prediction and `Send + Sync`.

use std::sync::Arc;

use ndarray::Array2;
use rayon::ThreadPoolBuilder;
use sciencekit_common::{SKExecutionMode, SKOutlierLabel, SKOutlierPredictor};
use sprs::{CsMat, TriMat};

use crate::{
    SKContamination, SKIsolationForest, SKIsolationForestBuilder, SKIsolationForestModel,
    SKIsolationMaximumSamples,
};

fn build_fixture(rows: usize) -> (Array2<f64>, CsMat<f64>) {
    let dense = Array2::from_shape_fn((rows, 4), |(row, column)| {
        if row + 4 >= rows {
            // Planted extreme outliers at the tail.
            35.0 * ((column as f64) + 1.0) * if row % 2 == 0 { 1.0 } else { -1.0 }
        } else if (row + column) % 3 == 0 {
            (((row * 17 + column * 11) % 31) as f64 - 15.0) * 0.05
        } else {
            0.0
        }
    });
    let mut triplets = TriMat::new((rows, 4));
    dense
        .rows()
        .into_iter()
        .enumerate()
        .for_each(|(row_index, values)| {
            values
                .iter()
                .enumerate()
                .filter(|&(_, &value)| value != 0.0)
                .for_each(|(column_index, &value)| {
                    triplets.add_triplet(row_index, column_index, value);
                });
        });
    (dense, triplets.to_csr())
}

#[test]
fn isolation_concurrency_types_are_send_and_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SKOutlierLabel>();
    assert_send_sync::<SKIsolationForestBuilder<f64>>();
    assert_send_sync::<SKIsolationForestBuilder<f32>>();
    assert_send_sync::<SKIsolationForest<f64>>();
    assert_send_sync::<SKIsolationForest<f32>>();
    assert_send_sync::<SKIsolationForestModel<f64>>();
    assert_send_sync::<SKIsolationForestModel<f32>>();
}

#[test]
fn isolation_concurrency_repeated_fits_and_thread_counts_are_bit_identical() {
    let (dense, csr) = build_fixture(320);
    let synchronous_detector = SKIsolationForest::<f64>::build()
        .number_of_trees(48)
        .maximum_samples(SKIsolationMaximumSamples::Count(128))
        .contamination(SKContamination::Fraction(0.05))
        .maximum_features(0.75)
        .random_seed(2026)
        .execution_mode(SKExecutionMode::InProcessSynchronous)
        .build()
        .unwrap();
    let automatic_detector = SKIsolationForest::<f64>::build()
        .number_of_trees(48)
        .maximum_samples(SKIsolationMaximumSamples::Count(128))
        .contamination(SKContamination::Fraction(0.05))
        .maximum_features(0.75)
        .random_seed(2026)
        .execution_mode(SKExecutionMode::Automatic)
        .build()
        .unwrap();

    let first_model = synchronous_detector.fit(&dense).unwrap();
    let second_model = synchronous_detector.fit(&dense).unwrap();
    assert_eq!(first_model, second_model);

    let baseline_scores = first_model.score_samples(&dense).unwrap();
    let baseline_decisions = first_model.decision_function(&dense).unwrap();
    let baseline_labels = first_model.predict(&dense).unwrap();

    for threads in [1usize, 2, 4, 8] {
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let (dense_model, sparse_model, scores, decisions, labels) = pool.install(|| {
            let dense_fitted = automatic_detector.fit(&dense).unwrap();
            let sparse_fitted = automatic_detector.fit(csr.view()).unwrap();
            let pool_scores = dense_fitted.score_samples(&dense).unwrap();
            let pool_decisions = dense_fitted.decision_function(csr.view()).unwrap();
            let pool_labels = sparse_fitted.predict(csr.view()).unwrap();
            (
                dense_fitted,
                sparse_fitted,
                pool_scores,
                pool_decisions,
                pool_labels,
            )
        });

        assert_eq!(dense_model.offset(), first_model.offset());
        assert_eq!(sparse_model.offset(), first_model.offset());
        (0..48).for_each(|tree_index| {
            assert_eq!(dense_model.tree(tree_index), first_model.tree(tree_index));
            assert_eq!(sparse_model.tree(tree_index), first_model.tree(tree_index));
        });
        assert_eq!(scores, baseline_scores);
        assert_eq!(decisions, baseline_decisions);
        assert_eq!(labels, baseline_labels);
    }
}

#[test]
fn isolation_concurrency_shared_model_serves_parallel_callers_lock_free() {
    let (dense, csr) = build_fixture(512);
    let dense = Arc::new(dense);
    let csr = Arc::new(csr);

    let model = Arc::new(
        SKIsolationForest::<f64>::build()
            .number_of_trees(32)
            .contamination(SKContamination::Fraction(4.0 / 512.0))
            .random_seed(77)
            .build()
            .unwrap()
            .fit(dense.view())
            .unwrap(),
    );

    let expected_scores = model.score_samples(dense.view()).unwrap();
    let expected_decisions = model.decision_function(dense.view()).unwrap();
    let expected_labels = model.predict(dense.view()).unwrap();

    let handles: Vec<_> = (0..8)
        .map(|worker_id| {
            let model = Arc::clone(&model);
            let dense = Arc::clone(&dense);
            let csr = Arc::clone(&csr);
            std::thread::spawn(move || {
                if worker_id % 2 == 0 {
                    (
                        SKOutlierPredictor::score_samples(&*model, dense.view()).unwrap(),
                        SKOutlierPredictor::decision_function(&*model, dense.view()).unwrap(),
                        SKOutlierPredictor::predict(&*model, dense.view()).unwrap(),
                    )
                } else {
                    (
                        SKOutlierPredictor::score_samples(&*model, csr.view()).unwrap(),
                        SKOutlierPredictor::decision_function(&*model, csr.view()).unwrap(),
                        SKOutlierPredictor::predict(&*model, csr.view()).unwrap(),
                    )
                }
            })
        })
        .collect();

    handles.into_iter().for_each(|handle| {
        let (scores, decisions, labels) = handle.join().unwrap();
        assert_eq!(scores, expected_scores);
        assert_eq!(decisions, expected_decisions);
        assert_eq!(labels, expected_labels);
    });
}
