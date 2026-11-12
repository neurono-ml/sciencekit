//! Companion tests for `SKIncrementalPCA::fit_streaming` over one-shot and
//! reopenable `SKSourceProvider` sources (spec `incremental-pca`).

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::{Array2, Axis};
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKDataBatch, SKError, SKLazySource, SKReopenableSource,
    SKUnsupervisedFit,
};
use sprs::{CsMat, TriMat};

use super::SKIncrementalPCA;
use crate::pca::test_tracing::PcaCaptureGuard;

struct BatchQueueSource<F> {
    batches: VecDeque<SKDataBatch<F>>,
}

impl<F> BatchQueueSource<F> {
    fn from_batches(batches: Vec<SKDataBatch<F>>) -> Self {
        Self {
            batches: batches.into(),
        }
    }
}

impl<F: Send> SKLazySource<F> for BatchQueueSource<F> {
    type Error = SKError;

    fn next_batch(&mut self) -> Result<Option<SKDataBatch<F>>, Self::Error> {
        Ok(self.batches.pop_front())
    }
}

fn csr_from_dense(dense: &Array2<f64>) -> CsMat<f64> {
    let mut triplet = TriMat::new(dense.dim());
    for ((row, column), &value) in dense.indexed_iter() {
        if value != 0.0 {
            triplet.add_triplet(row, column, value);
        }
    }
    triplet.to_csr()
}

fn split_dense_batches(data: &Array2<f64>, chunk_rows: usize) -> Vec<SKDataBatch<f64>> {
    let chunks: Vec<Array2<f64>> = data
        .axis_chunks_iter(Axis(0), chunk_rows)
        .map(|chunk| chunk.to_owned())
        .collect();
    let total_chunks = chunks.len();
    chunks
        .into_iter()
        .enumerate()
        .map(|(index, chunk)| SKDataBatch::new_dense(chunk, index, index + 1 == total_chunks))
        .collect()
}

#[test]
fn incremental_streaming_one_shot_provider_consumes_one_pass_and_matches_fit() {
    let data = Array2::from_shape_fn((44, 5), |(row, column)| {
        let angle = (row as f64 + 1.0) * 0.17 + (column as f64) * 0.43;
        angle.cos() * (6.0 - column as f64) + ((row * 5 + column * 7) % 11) as f64 * 0.25
    });
    let mut one_shot = BatchQueueSource::from_batches(split_dense_batches(&data, 12));

    let estimator = SKIncrementalPCA::<f64>::build()
        .number_of_components(3)
        .maximum_number_of_epochs(4)
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap();

    let capture = PcaCaptureGuard::begin();
    let streamed = estimator.fit_streaming(&mut one_shot).unwrap();
    let warnings = capture.warning_events();
    let spans = capture.spans();

    assert_eq!(streamed.epochs_consumed(), 1);
    assert_eq!(streamed.samples_seen(), 44);
    assert!(!warnings.is_empty());
    assert!(spans.iter().any(|(name, fields)| {
        name == "sciencekit.operation"
            && fields.get("operation").map(String::as_str) == Some("partial_fit")
            && fields.get("backend").map(String::as_str) == Some("matrixmultiply")
    }));

    let in_memory = estimator.fit(&data).unwrap();
    for (left, right) in streamed
        .components()
        .iter()
        .zip(in_memory.components().iter())
    {
        assert!((left - right).abs() < 1e-9, "left={left}, right={right}");
    }
    for (left, right) in streamed
        .explained_variance()
        .iter()
        .zip(in_memory.explained_variance().iter())
    {
        assert!((left - right).abs() < 1e-9, "left={left}, right={right}");
    }
}

#[test]
fn incremental_streaming_reopenable_multi_epoch_reopens_and_preserves_subspace() {
    let data = Array2::from_shape_fn((36, 5), |(row, column)| {
        let angle = (row as f64 + 1.0) * 0.29;
        angle.sin() * (column as f64 + 1.5) * 3.0 + ((row * 3 + column * 5) % 9) as f64 * 0.4
    });
    let batches = split_dense_batches(&data, 12);
    let open_calls = Arc::new(AtomicUsize::new(0));
    let open_calls_counter = Arc::clone(&open_calls);

    let provider = SKReopenableSource::new(move || {
        open_calls_counter.fetch_add(1, Ordering::SeqCst);
        Ok(BatchQueueSource::from_batches(batches.clone()))
    });

    let estimator = SKIncrementalPCA::<f64>::build()
        .number_of_components(3)
        .maximum_number_of_epochs(3)
        .build()
        .unwrap();

    let multi_epoch = estimator.fit_streaming(provider).unwrap();
    let single_pass = estimator.fit(&data).unwrap();

    assert_eq!(open_calls.load(Ordering::SeqCst), 3);
    assert_eq!(multi_epoch.epochs_consumed(), 3);
    assert_eq!(multi_epoch.samples_seen(), 108);

    for (left, right) in multi_epoch.mean().iter().zip(single_pass.mean().iter()) {
        assert!((left - right).abs() < 1e-11);
    }
    for (left, right) in multi_epoch
        .components()
        .iter()
        .zip(single_pass.components().iter())
    {
        assert!((left - right).abs() < 1e-9);
    }
    for (left, right) in multi_epoch
        .explained_variance_ratio()
        .iter()
        .zip(single_pass.explained_variance_ratio().iter())
    {
        assert!((left - right).abs() < 1e-10);
    }
}

#[test]
fn incremental_streaming_sparse_and_mixed_batches_match_dense_fit() {
    let dense = Array2::from_shape_fn((40, 6), |(row, column)| {
        if (row * 5 + column * 3) % 4 == 0 {
            let angle = (row as f64 + 1.0) * 0.31 + (column as f64) * 0.47;
            angle.sin() * (7.0 - column as f64) + 0.5
        } else {
            0.0
        }
    });
    let batches: Vec<SKDataBatch<f64>> = dense
        .axis_chunks_iter(Axis(0), 10)
        .enumerate()
        .map(|(index, chunk)| {
            let owned = chunk.to_owned();
            if index % 2 == 0 {
                SKDataBatch::new_sparse(csr_from_dense(&owned), index, index == 3)
            } else {
                SKDataBatch::new_dense(owned, index, index == 3)
            }
        })
        .collect();
    let mut source = BatchQueueSource::from_batches(batches);

    let estimator = SKIncrementalPCA::<f64>::build()
        .number_of_components(4)
        .build()
        .unwrap();
    let streamed = estimator.fit_streaming(&mut source).unwrap();
    let dense_fit = estimator.fit(&dense).unwrap();

    assert_eq!(streamed.samples_seen(), 40);
    for (left, right) in streamed
        .components()
        .iter()
        .zip(dense_fit.components().iter())
    {
        assert!((left - right).abs() < 1e-9);
    }
}
