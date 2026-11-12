//! Companion tests for `Send + Sync` concurrent prediction and large-batch
//! parallel-equals-sequential equivalence (spec `support-vector-machines`).

use std::sync::Arc;
use std::thread;

use ndarray::{Array1, Array2};
use sciencekit_common::{
    SKBuilder, SKClassifierPredictor, SKExecutionMode, SKRegressorPredictor, SKSupervisedFit,
};
use sciencekit_math::SKKernel;

use crate::{SKSVC, SKSVCModel, SKSVR, SKSVRModel};

fn assert_send_and_sync<T: Send + Sync>() {}

#[test]
fn concurrency_shared_models_predict_identically_across_threads() {
    assert_send_and_sync::<SKSVC<f64>>();
    assert_send_and_sync::<SKSVCModel<f64>>();
    assert_send_and_sync::<SKSVR<f64>>();
    assert_send_and_sync::<SKSVRModel<f64>>();

    let train_x = Array2::from_shape_fn((30, 4), |(r, c)| {
        let sign = if r < 15 { -1.0 } else { 1.0 };
        sign * 2.0 + ((r * 3 + c * 7) % 11) as f64 * 0.05
    });
    let train_labels = Array1::from_shape_fn(30, |r| i64::from(r >= 15));
    let train_targets = Array1::from_shape_fn(30, |r| train_x[[r, 0]] * 1.5 - 0.4);

    let svc = Arc::new(
        SKSVC::<f64>::build()
            .kernel(SKKernel::RadialBasisFunction)
            .probability(true)
            .random_seed(9)
            .build()
            .unwrap()
            .fit(train_x.view(), train_labels.view())
            .unwrap(),
    );
    let svr = Arc::new(
        SKSVR::<f64>::build()
            .kernel(SKKernel::RadialBasisFunction)
            .build()
            .unwrap()
            .fit(train_x.view(), train_targets.view())
            .unwrap(),
    );

    let query = Array2::from_shape_fn((512, 4), |(r, c)| ((r * 5 + c * 3) % 23) as f64 * 0.2 - 2.0);
    let ref_labels = svc.predict_labels(query.view()).unwrap();
    let ref_probs = svc.predict_probabilities(query.view()).unwrap();
    let ref_reg = svr.predict(query.view()).unwrap();

    thread::scope(|scope| {
        let handles: Vec<_> = (0..6)
            .map(|_| {
                let svc_ref = Arc::clone(&svc);
                let svr_ref = Arc::clone(&svr);
                let q = query.view();
                scope.spawn(move || {
                    (
                        svc_ref.predict_labels(q).unwrap(),
                        svc_ref.predict_probabilities(q).unwrap(),
                        svr_ref.predict(q).unwrap(),
                    )
                })
            })
            .collect();

        for handle in handles {
            let (labels, probs, reg) = handle.join().unwrap();
            assert_eq!(labels, ref_labels);
            assert_eq!(probs, ref_probs);
            assert_eq!(reg, ref_reg);
        }
    });
}

#[test]
fn concurrency_large_batch_parallel_prediction_matches_single_row_sequential() {
    let train_x = Array2::from_shape_fn((36, 4), |(r, c)| {
        let cluster = (r / 12) as f64 * 3.0 - 3.0;
        cluster + ((r * 7 + c * 5) % 9) as f64 * 0.08
    });
    let train_labels = Array1::from_shape_fn(36, |r| (r / 12) as i64);
    let train_targets = Array1::from_shape_fn(36, |r| train_x[[r, 0]] * 0.8 + train_x[[r, 1]]);

    let svc = SKSVC::<f64>::build()
        .kernel(SKKernel::RadialBasisFunction)
        .execution_mode(SKExecutionMode::Automatic)
        .build()
        .unwrap()
        .fit(train_x.view(), train_labels.view())
        .unwrap();
    let svr = SKSVR::<f64>::build()
        .kernel(SKKernel::RadialBasisFunction)
        .execution_mode(SKExecutionMode::Automatic)
        .build()
        .unwrap()
        .fit(train_x.view(), train_targets.view())
        .unwrap();

    // 600 query rows * 4 features = 2400 > 1024 grain & >= 256 rows:
    // 4-thread pool runs the row-parallel batch path; 1-thread pool runs the sequential batch path.
    let query = Array2::from_shape_fn((600, 4), |(r, c)| {
        ((r * 11 + c * 3) % 29) as f64 * 0.25 - 3.5
    });
    let pool_parallel = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let pool_sequential = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();

    let (batch_labels, batch_reg) = pool_parallel.install(|| {
        (
            svc.predict_labels(query.view()).unwrap(),
            svr.predict(query.view()).unwrap(),
        )
    });
    let (seq_batch_labels, seq_batch_reg) = pool_sequential.install(|| {
        (
            svc.predict_labels(query.view()).unwrap(),
            svr.predict(query.view()).unwrap(),
        )
    });
    assert_eq!(batch_labels, seq_batch_labels);
    assert_eq!(batch_reg, seq_batch_reg);

    for row in 0..query.nrows() {
        let single_view = query.row(row).insert_axis(ndarray::Axis(0));
        assert_eq!(
            svc.predict_labels(single_view).unwrap()[0],
            batch_labels[row]
        );
        assert!((svr.predict(single_view).unwrap()[0] - batch_reg[row]).abs() < 1e-12);
    }
}
