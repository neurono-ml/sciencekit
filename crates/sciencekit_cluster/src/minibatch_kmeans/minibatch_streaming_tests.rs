//! Companion tests for `fit_streaming` over one-shot and reopenable providers
//! (spec `mini-batch-k-means`, tasks 3.1 and 3.2).

use std::collections::VecDeque;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use ndarray::{Array2, array};
use sciencekit_common::{SKBuilder, SKDataBatch, SKError, SKLazySource, SKReopenableSource};
use sprs::TriMat;

use super::SKMiniBatchKMeans;

struct VecBatchSource {
    batches: VecDeque<SKDataBatch<f64>>,
}

impl VecBatchSource {
    fn from_dense(chunks: Vec<Array2<f64>>) -> Self {
        let total = chunks.len();
        let batches = chunks
            .into_iter()
            .enumerate()
            .map(|(index, chunk)| SKDataBatch::new_dense(chunk, index, index + 1 == total))
            .collect();
        Self { batches }
    }

    fn from_batches(batches: Vec<SKDataBatch<f64>>) -> Self {
        Self {
            batches: batches.into_iter().collect(),
        }
    }
}

impl SKLazySource<f64> for VecBatchSource {
    type Error = SKError;

    fn next_batch(&mut self) -> Result<Option<SKDataBatch<f64>>, Self::Error> {
        Ok(self.batches.pop_front())
    }
}

fn three_planted_chunks() -> Vec<Array2<f64>> {
    vec![
        array![
            [0.0_f64, 0.0],
            [0.2, 0.1],
            [-0.1, 0.2],
            [30.0, 30.0],
            [30.1, 29.9],
            [29.8, 30.2]
        ],
        array![
            [0.1_f64, -0.1],
            [-0.2, 0.0],
            [0.0, 0.3],
            [30.2, 30.1],
            [29.9, 29.8],
            [30.0, 30.1]
        ],
        array![
            [0.0_f64, 0.2],
            [0.1, 0.0],
            [-0.1, -0.1],
            [29.9, 30.0],
            [30.1, 30.2],
            [30.0, 29.9]
        ],
    ]
}

#[test]
fn minibatch_streaming_one_shot_provider_consumes_single_pass_and_matches_partial_fit() {
    let chunks = three_planted_chunks();
    let estimator = SKMiniBatchKMeans::<f64>::build()
        .number_of_clusters(2)
        .maximum_number_of_epochs(4)
        .random_seed(31)
        .build()
        .unwrap();

    let mut one_shot = VecBatchSource::from_dense(chunks.clone());
    let streamed = estimator.fit_streaming(&mut one_shot).unwrap();

    assert_eq!(streamed.epochs_consumed(), 1);
    assert_eq!(streamed.batches_processed(), 3);
    assert_eq!(streamed.counts().sum(), 18);
    assert!(streamed.labels().is_none());

    let mut manual = estimator.fit_batch(&chunks[0]).unwrap();
    manual
        .partial_fit(&chunks[1])
        .unwrap()
        .partial_fit(&chunks[2])
        .unwrap();

    assert_eq!(streamed.counts(), manual.counts());
    assert!((streamed.inertia() - manual.inertia()).abs() < 1e-12);
    streamed
        .cluster_centers()
        .iter()
        .zip(manual.cluster_centers().iter())
        .for_each(|(&left, &right)| assert!((left - right).abs() < 1e-12));
}

#[test]
fn minibatch_streaming_reopenable_provider_runs_multiple_epochs_and_sparse_batches() {
    let opens = Arc::new(AtomicUsize::new(0));
    let opens_counter = Arc::clone(&opens);
    let provider = SKReopenableSource::new(move || {
        opens_counter.fetch_add(1, Ordering::SeqCst);
        let chunks = three_planted_chunks();
        let total = chunks.len();
        let sparse_batches = chunks
            .into_iter()
            .enumerate()
            .map(|(index, dense)| {
                let mut triplet = TriMat::new(dense.dim());
                dense
                    .indexed_iter()
                    .filter(|(_, value)| **value != 0.0)
                    .for_each(|((row, col), &value)| triplet.add_triplet(row, col, value));
                SKDataBatch::new_sparse(triplet.to_csr(), index, index + 1 == total)
            })
            .collect();
        Ok(VecBatchSource::from_batches(sparse_batches))
    });

    let estimator = SKMiniBatchKMeans::<f64>::build()
        .number_of_clusters(2)
        .maximum_number_of_epochs(3)
        .random_seed(7)
        .build()
        .unwrap();

    let model = estimator.fit_streaming(provider).unwrap();
    assert_eq!(opens.load(Ordering::SeqCst), 3);
    assert_eq!(model.epochs_consumed(), 3);
    assert_eq!(model.batches_processed(), 9);
    assert_eq!(model.counts().sum(), 54);
    assert!(!model.converged());
}
