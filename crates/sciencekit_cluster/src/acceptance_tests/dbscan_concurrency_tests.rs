//! Concurrency acceptance scenarios for `SKDBSCAN`.

use ndarray::{Axis, Slice};
use sciencekit_common::{SKBuilder, SKClustererPredictor, SKExecutionMode, SKUnsupervisedFit};

use super::dbscan_fixtures::planted_density_chains;
use crate::{SKDBSCAN, SKDBSCANModel};

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn dbscan_acceptance_concurrency_shared_model_and_estimator_serve_parallel_threads() {
    assert_send_sync::<SKDBSCAN<f64>>();
    assert_send_sync::<SKDBSCANModel<f64>>();

    let (data, _) = planted_density_chains(1_600, 4, 20, 40);
    let estimator = SKDBSCAN::<f64>::build()
        .eps(0.45)
        .minimum_samples(5)
        .execution_mode(SKExecutionMode::Automatic)
        .build()
        .unwrap();
    let model = estimator.fit(&data).unwrap();
    let expected = model.predict_clusters(&data).unwrap();

    std::thread::scope(|scope| {
        let predict_handles: Vec<_> = (0..4)
            .map(|chunk| {
                let start = chunk * 410;
                let end = (chunk + 1) * 410;
                let batch = data.slice_axis(Axis(0), Slice::from(start..end)).to_owned();
                let model_ref = &model;
                scope.spawn(move || (start, end, model_ref.predict_clusters(&batch).unwrap()))
            })
            .collect();

        for handle in predict_handles {
            let (start, end, labels) = handle.join().expect("prediction thread joins");
            assert_eq!(
                labels,
                expected.slice_axis(Axis(0), Slice::from(start..end))
            );
        }

        let fit_handles: Vec<_> = [
            SKExecutionMode::Automatic,
            SKExecutionMode::InProcessSynchronous,
            SKExecutionMode::InProcessAsynchronous,
        ]
        .into_iter()
        .map(|mode| {
            let data_ref = &data;
            scope.spawn(move || {
                SKDBSCAN::<f64>::build()
                    .eps(0.45)
                    .minimum_samples(5)
                    .execution_mode(mode)
                    .build()
                    .unwrap()
                    .fit(data_ref)
                    .unwrap()
            })
        })
        .collect();

        for handle in fit_handles {
            let fitted = handle.join().expect("fit thread joins");
            assert_eq!(fitted.labels(), model.labels());
            assert_eq!(fitted.core_sample_indices(), model.core_sample_indices());
        }
    });
}
