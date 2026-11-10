//! Concurrency acceptance for the nearest-neighbor imputer.

use ndarray::{Array2, Axis, Slice};
use sciencekit_common::{SKBuilder, SKFeatureTransformer, SKUnsupervisedFit};

use super::fixtures::{dense, linear_design};
use crate::knn::core_implementation::SKKNNImputer;

#[test]
fn acceptance_knn_imputer_one_model_serves_several_threads() {
    let training = linear_design(600, 4);
    let model = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(5)
        .weights(crate::knn::SKImputationWeights::Distance)
        .add_indicator(true)
        .build()
        .unwrap()
        .fit(training.view())
        .unwrap();

    let mut probe = linear_design(320, 4);
    for row in (0..320).step_by(4) {
        probe[[row, 3]] = f64::NAN;
    }
    let expected = dense(model.transform(&probe).unwrap());
    let batch = 80;

    std::thread::scope(|scope| {
        for chunk in 0..4 {
            let model = &model;
            let expected = expected.clone();
            let block = probe
                .slice_axis(Axis(0), Slice::from(chunk * batch..(chunk + 1) * batch))
                .to_owned();
            scope.spawn(move || {
                let found = dense(model.transform(&block).unwrap());
                assert_eq!(
                    found,
                    expected.slice_axis(
                        Axis(0),
                        Slice::from(chunk * batch..(chunk + 1) * batch)
                    )
                );
            });
        }
    });
}

#[test]
fn acceptance_knn_imputer_concurrent_fits_share_the_estimator() {
    let training: Array2<f64> = linear_design(200, 3);
    let estimator = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(3)
        .build()
        .unwrap();
    let expected = estimator.fit(training.view()).unwrap().column_means().to_owned();

    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..8)
            .map(|_| {
                scope.spawn(|| {
                    estimator
                        .fit(training.view())
                        .unwrap()
                        .column_means()
                        .to_owned()
                })
            })
            .collect();
        for handle in handles {
            assert_eq!(handle.join().unwrap(), expected);
        }
    });
}
