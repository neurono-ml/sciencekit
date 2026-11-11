//! Determinism, thread-count invariance, `Send + Sync` and concurrency suites.

use std::sync::Arc;
use std::thread;

use ndarray::Array2;
use rayon::ThreadPoolBuilder;
use sciencekit_common::{
    SKBackendKind, SKBuilder, SKClustererPredictor, SKReopenableSource, SKUnsupervisedFit,
};

use super::super::{SKMiniBatchKMeans, SKMiniBatchKMeansBuilder, SKMiniBatchKMeansModel};
use super::fixtures::{
    BackendSpanRecorder, ChunkedSource, assert_send_and_sync, deterministic_dataset,
};

#[test]
fn minibatch_determinism_send_and_sync_bounds_hold_for_estimator_and_model() {
    assert_send_and_sync::<SKMiniBatchKMeansBuilder<f64>>();
    assert_send_and_sync::<SKMiniBatchKMeans<f64>>();
    assert_send_and_sync::<SKMiniBatchKMeansModel<f64>>();
    assert_send_and_sync::<SKMiniBatchKMeansModel<f32>>();
}

#[test]
fn minibatch_determinism_repeated_fits_and_streaming_runs_are_bit_identical() {
    let data = deterministic_dataset(180);
    let estimator = SKMiniBatchKMeans::<f64>::build()
        .number_of_clusters(3)
        .batch_size(32)
        .initialization_batch_size(64)
        .maximum_number_of_iterations(8)
        .maximum_number_of_epochs(2)
        .reassignment_ratio(0.05)
        .random_seed(123)
        .build()
        .unwrap();

    assert_eq!(estimator.fit(&data).unwrap(), estimator.fit(&data).unwrap());

    let make_provider = |matrix: Array2<f64>| {
        SKReopenableSource::new(move || {
            Ok(ChunkedSource {
                data: matrix.clone(),
                cursor: 0,
                chunk_size: 32,
            })
        })
    };
    assert_eq!(
        estimator.fit_streaming(make_provider(data.clone())).unwrap(),
        estimator.fit_streaming(make_provider(data)).unwrap()
    );
}

#[test]
fn minibatch_determinism_thread_count_invariance_between_one_and_four_rayon_threads() {
    let data = deterministic_dataset(240);
    let estimator = SKMiniBatchKMeans::<f64>::build()
        .number_of_clusters(3)
        .batch_size(40)
        .initialization_batch_size(80)
        .maximum_number_of_iterations(6)
        .random_seed(88)
        .build()
        .unwrap();

    let pool_one = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
    let pool_four = ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    let single_thread = pool_one.install(|| estimator.fit(&data).unwrap());
    let multi_thread = pool_four.install(|| estimator.fit(&data).unwrap());
    assert_eq!(single_thread, multi_thread);
}

#[test]
fn minibatch_determinism_concurrent_fits_and_predictions_plus_backend_spans() {
    let data = Arc::new(deterministic_dataset(120));
    let estimator = Arc::new(
        SKMiniBatchKMeans::<f64>::build()
            .number_of_clusters(3)
            .batch_size(30)
            .maximum_number_of_iterations(5)
            .backend(SKBackendKind::MatrixMultiply)
            .random_seed(19)
            .build()
            .unwrap(),
    );
    let recorder = Arc::new(BackendSpanRecorder::default());
    let reference = tracing::subscriber::with_default(Arc::clone(&recorder), || {
        let mut model = estimator.fit(data.as_ref()).unwrap();
        let _ = estimator.fit_batch(data.as_ref()).unwrap();
        let _ = estimator.fit_predict(data.as_ref()).unwrap();
        let mut stream = ChunkedSource {
            data: data.as_ref().clone(),
            cursor: 0,
            chunk_size: 40,
        };
        let _ = estimator.fit_streaming(&mut stream).unwrap();
        model.partial_fit(data.as_ref()).unwrap();
        let _ = model.predict_clusters(data.as_ref()).unwrap();
        let _ = model.transform(data.as_ref()).unwrap();
        Arc::new(model)
    });
    let recorded = recorder.backends.lock().unwrap().clone();
    assert!(recorded.len() >= 7);
    assert!(recorded.iter().all(|name| name == "matrixmultiply"));

    let expected_fit = estimator.fit(data.as_ref()).unwrap();
    let expected_labels = reference.predict_clusters(data.as_ref()).unwrap();
    let expected_distances = reference.transform(data.as_ref()).unwrap();
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let shared_data = Arc::clone(&data);
            let shared_estimator = Arc::clone(&estimator);
            let shared_model = Arc::clone(&reference);
            thread::spawn(move || {
                (
                    shared_estimator.fit(shared_data.as_ref()).unwrap(),
                    shared_model.predict_clusters(shared_data.as_ref()).unwrap(),
                    shared_model.transform(shared_data.as_ref()).unwrap(),
                )
            })
        })
        .collect();
    handles.into_iter().for_each(|handle| {
        let (fitted, labels, distances) = handle.join().unwrap();
        assert_eq!(fitted, expected_fit);
        assert_eq!(labels, expected_labels);
        assert_eq!(distances, expected_distances);
    });
}
