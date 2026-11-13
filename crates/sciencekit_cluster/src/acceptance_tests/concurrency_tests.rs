//! Concurrency acceptance scenarios.

use ndarray::{Axis, Slice};
use sciencekit_common::{SKBuilder, SKClustererPredictor, SKUnsupervisedFit};

use super::fixtures::planted_design;
use crate::{SKKMeans, SKKMeansModel};

fn assert_send_sync<T: Send + Sync>() {}

#[test]
fn acceptance_concurrency_estimator_and_model_are_send_and_sync() {
    assert_send_sync::<SKKMeans<f64>>();
    assert_send_sync::<SKKMeansModel<f64>>();
}

#[test]
fn acceptance_concurrency_shared_model_serves_parallel_assignments() {
    let (data, _) = planted_design(4_000, 4);
    let model = SKKMeans::<f64>::build()
        .number_of_clusters(4)
        .random_seed(31)
        .build()
        .unwrap()
        .fit(&data)
        .unwrap();
    let expected = model.predict_clusters(&data).unwrap();

    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for chunk in 0..4 {
            let batch = data
                .slice_axis(Axis(0), Slice::from(chunk * 1_000..(chunk + 1) * 1_000))
                .to_owned();
            let model = &model;
            handles.push(scope.spawn(move || (chunk, model.predict_clusters(&batch).unwrap())));
        }
        for handle in handles {
            let (chunk, labels) = handle.join().expect("thread joins");
            let range = Slice::from(chunk * 1_000..(chunk + 1) * 1_000);
            assert_eq!(labels, expected.slice_axis(Axis(0), range));
        }
    });
}

#[test]
fn acceptance_concurrency_parallel_fits_agree_with_the_sequential_fit() {
    let (data, _) = planted_design(1_000, 3);
    let estimator = SKKMeans::<f64>::build()
        .number_of_clusters(3)
        .random_seed(8)
        .build()
        .unwrap();
    let expected = estimator.fit(&data).unwrap();

    std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for _ in 0..3 {
            let estimator = &estimator;
            let data = &data;
            handles.push(scope.spawn(move || estimator.fit(data).unwrap()));
        }
        for handle in handles {
            let model = handle.join().expect("thread joins");
            assert_eq!(model.cluster_centers(), expected.cluster_centers());
            assert_eq!(model.labels(), expected.labels());
            assert_eq!(model.inertia(), expected.inertia());
        }
    });
}
