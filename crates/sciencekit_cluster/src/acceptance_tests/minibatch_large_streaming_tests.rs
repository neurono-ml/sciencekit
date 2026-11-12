//! Large 50 000-row streaming, reopenable multi-epoch and dense/CSR parity
//! acceptance scenarios for `SKMiniBatchKMeans` (spec `mini-batch-k-means`,
//! task 5.1).

use std::sync::Arc;

use ndarray::{Array2, Axis, Slice};
use sciencekit_common::{
    SKBuilder, SKClustererPredictor, SKDataBatch, SKError, SKLazySource, SKReopenableSource,
    SKUnsupervisedFit,
};
use sprs::{CsMat, TriMat};

use super::fixtures::{csr_from_dense, majority_accuracy, planted_design};
use crate::SKMiniBatchKMeans;

struct DenseChunkStream {
    data: Arc<Array2<f64>>,
    cursor: usize,
    chunk_rows: usize,
}

impl SKLazySource<f64> for DenseChunkStream {
    type Error = SKError;

    fn next_batch(&mut self) -> Result<Option<SKDataBatch<f64>>, Self::Error> {
        if self.cursor >= self.data.nrows() {
            return Ok(None);
        }
        let end = (self.cursor + self.chunk_rows).min(self.data.nrows());
        let block = self
            .data
            .slice_axis(Axis(0), Slice::from(self.cursor..end))
            .to_owned();
        let position = self.cursor / self.chunk_rows;
        self.cursor = end;
        Ok(Some(SKDataBatch::new_dense(
            block,
            position,
            self.cursor >= self.data.nrows(),
        )))
    }
}

struct SparseChunkStream {
    data: Arc<Array2<f64>>,
    cursor: usize,
    chunk_rows: usize,
}

impl SKLazySource<f64> for SparseChunkStream {
    type Error = SKError;

    fn next_batch(&mut self) -> Result<Option<SKDataBatch<f64>>, Self::Error> {
        if self.cursor >= self.data.nrows() {
            return Ok(None);
        }
        let end = (self.cursor + self.chunk_rows).min(self.data.nrows());
        let slice = self
            .data
            .slice_axis(Axis(0), Slice::from(self.cursor..end))
            .to_owned();
        let sparse: CsMat<f64> = csr_from_dense(&slice);
        let position = self.cursor / self.chunk_rows;
        self.cursor = end;
        Ok(Some(SKDataBatch::new_sparse(
            sparse,
            position,
            self.cursor >= self.data.nrows(),
        )))
    }
}

#[test]
fn minibatch_acceptance_fifty_thousand_row_one_shot_and_multi_epoch_streaming_parity() {
    let (raw_data, truth) = planted_design(50_000, 4);
    let shared_data = Arc::new(raw_data);

    let estimator = SKMiniBatchKMeans::<f64>::build()
        .number_of_clusters(4)
        .batch_size(1_000)
        .initialization_batch_size(1_000)
        .maximum_number_of_epochs(2)
        .tolerance(0.0)
        .random_seed(0)
        .build()
        .unwrap();

    // 1. One-shot provider: consumes a single pass of 50,000 rows even when 2 epochs are requested.
    let mut one_shot_dense = DenseChunkStream {
        data: Arc::clone(&shared_data),
        cursor: 0,
        chunk_rows: 1_000,
    };
    let one_pass_model = estimator.fit_streaming(&mut one_shot_dense).unwrap();
    assert_eq!(one_pass_model.epochs_consumed(), 1);
    assert_eq!(one_pass_model.batches_processed(), 50);
    assert_eq!(one_pass_model.counts().sum(), 50_000);
    assert!(one_pass_model.labels().is_none());

    let one_pass_labels = one_pass_model
        .predict_clusters(shared_data.as_ref())
        .unwrap();
    assert_eq!(majority_accuracy(&truth, &one_pass_labels, 4), 1.0);

    // 2. Sparse one-shot stream over the same 50,000 rows matches the dense stream without densifying.
    let mut one_shot_sparse = SparseChunkStream {
        data: Arc::clone(&shared_data),
        cursor: 0,
        chunk_rows: 1_000,
    };
    let sparse_stream_model = estimator.fit_streaming(&mut one_shot_sparse).unwrap();
    assert_eq!(sparse_stream_model.counts(), one_pass_model.counts());
    sparse_stream_model
        .cluster_centers()
        .iter()
        .zip(one_pass_model.cluster_centers().iter())
        .for_each(|(&left, &right)| assert!((left - right).abs() < 1e-8));

    // 3. Reopenable multi-epoch provider consumes 2 epochs (100,000 streamed row updates).
    let reopenable_data = Arc::clone(&shared_data);
    let multi_epoch_provider = SKReopenableSource::new(move || {
        Ok(DenseChunkStream {
            data: Arc::clone(&reopenable_data),
            cursor: 0,
            chunk_rows: 1_000,
        })
    });
    let multi_epoch_model = estimator.fit_streaming(multi_epoch_provider).unwrap();
    assert_eq!(multi_epoch_model.epochs_consumed(), 2);
    assert_eq!(multi_epoch_model.batches_processed(), 100);
    assert_eq!(multi_epoch_model.counts().sum(), 100_000);

    let multi_labels = multi_epoch_model
        .predict_clusters(shared_data.as_ref())
        .unwrap();
    assert_eq!(majority_accuracy(&truth, &multi_labels, 4), 1.0);
}

#[test]
fn minibatch_acceptance_wide_sparse_design_never_densifies() {
    let width = 1_000_000;
    let mut triplet = TriMat::new((4, width));
    triplet.add_triplet(0, 0, 10.0);
    triplet.add_triplet(1, 0, 10.2);
    triplet.add_triplet(2, width - 1, -10.0);
    triplet.add_triplet(3, width - 1, -10.2);
    let sparse = triplet.to_csr();
    let nnz_before = sparse.nnz();

    let model = SKMiniBatchKMeans::<f64>::build()
        .number_of_clusters(2)
        .batch_size(4)
        .maximum_number_of_iterations(4)
        .random_seed(0)
        .build()
        .unwrap()
        .fit(sparse.view())
        .unwrap();

    assert_eq!(model.number_of_features(), width);
    assert_eq!(sparse.nnz(), nnz_before);
    let labels = model.labels().unwrap();
    assert_eq!(labels[0], labels[1]);
    assert_eq!(labels[2], labels[3]);
    assert_ne!(labels[0], labels[2]);
}
