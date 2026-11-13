//! End-to-end acceptance tests for `SKIncrementalPCA` streaming passes,
//! `SKModelState` export/import, concurrency and `sciencekit_metrics` (PRD §8.7).

use std::collections::VecDeque;
use std::sync::Arc;

use ndarray::{Array1, Axis};
use sciencekit_common::{
    SKBuilder, SKDataBatch, SKError, SKExportable, SKImportable, SKLazySource, SKReopenableSource,
    SKUnsupervisedFit,
};
use sciencekit_metrics::regression::{sk_mean_squared_error, sk_r2_score};

use super::fixtures::planted_rank_matrix;
use crate::{SKIncrementalPCA, SKIncrementalPCAModel};

struct VecBatchSource<F> {
    batches: VecDeque<SKDataBatch<F>>,
}

impl<F: Send> SKLazySource<F> for VecBatchSource<F> {
    type Error = SKError;

    fn next_batch(&mut self) -> Result<Option<SKDataBatch<F>>, Self::Error> {
        Ok(self.batches.pop_front())
    }
}

#[test]
fn acceptance_incremental_pca_streaming_epochs_export_concurrency_and_metrics() {
    let data = planted_rank_matrix(640, 16, 3, 1e-3);
    let batches: Vec<SKDataBatch<f64>> = data
        .axis_chunks_iter(Axis(0), 160)
        .enumerate()
        .map(|(index, chunk)| SKDataBatch::new_dense(chunk.to_owned(), index, index == 3))
        .collect();

    let mut one_shot = VecBatchSource {
        batches: batches.clone().into(),
    };
    let one_pass = SKIncrementalPCA::<f64>::build()
        .number_of_components(3)
        .maximum_number_of_epochs(3)
        .whiten(true)
        .build()
        .unwrap()
        .fit_streaming(&mut one_shot)
        .unwrap();
    assert_eq!(one_pass.epochs_consumed(), 1);

    let reopenable_batches = batches.clone();
    let reopenable = SKReopenableSource::new(move || {
        Ok(VecBatchSource {
            batches: reopenable_batches.clone().into(),
        })
    });
    let multi_pass = SKIncrementalPCA::<f64>::build()
        .number_of_components(3)
        .maximum_number_of_epochs(3)
        .whiten(true)
        .build()
        .unwrap()
        .fit_streaming(reopenable)
        .unwrap();
    assert_eq!(multi_pass.epochs_consumed(), 3);
    assert_eq!(multi_pass.samples_seen(), 1_920);
    for (left, right) in one_pass
        .components()
        .iter()
        .zip(multi_pass.components().iter())
    {
        assert!((left - right).abs() < 1e-9);
    }

    // Export and import round-trip, then concurrent sharing + continued partial_fit.
    let exported = one_pass.export_state().unwrap();
    let mut imported = SKIncrementalPCAModel::<f64>::import_state(&exported).unwrap();
    let reference_scores = one_pass.transform(&data).unwrap();
    assert_eq!(imported.transform(&data).unwrap(), reference_scores);

    let shared = Arc::new(imported.clone());
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..6)
            .map(|_| {
                let model = Arc::clone(&shared);
                let data_ref = &data;
                scope.spawn(move || model.transform(data_ref).unwrap())
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), reference_scores);
        }
    });

    imported
        .partial_fit(data.slice_axis(Axis(0), ndarray::Slice::from(0..160)))
        .unwrap();
    assert_eq!(imported.samples_seen(), 800);

    // Reconstruction quality metrics via sciencekit_metrics.
    let flattened_truth = Array1::from_iter(data.iter().copied());
    let mut previous_error = f64::INFINITY;
    for components in [1usize, 2, 3] {
        let model = SKIncrementalPCA::<f64>::build()
            .number_of_components(components)
            .batch_size(160)
            .build()
            .unwrap()
            .fit(&data)
            .unwrap();
        let rebuilt = model
            .inverse_transform(&model.transform(&data).unwrap())
            .unwrap();
        let flattened_rebuilt = Array1::from_iter(rebuilt.iter().copied());
        let mean_squared_error =
            sk_mean_squared_error(flattened_truth.view(), flattened_rebuilt.view()).unwrap();
        let determination = sk_r2_score(flattened_truth.view(), flattened_rebuilt.view()).unwrap();
        assert!(mean_squared_error < previous_error);
        previous_error = mean_squared_error;
        if components == 3 {
            assert!(mean_squared_error < 1e-5);
            assert!(determination > 0.9999);
        }
    }
}
