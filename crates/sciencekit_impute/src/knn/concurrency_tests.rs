//! Companion tests for the concurrency contract
//! (spec `nearest-neighbor-imputation`).

use ndarray::{Array2, Axis, Slice};
use sciencekit_common::{SKBuilder, SKFeatureTransformer, SKUnsupervisedFit};

use crate::knn::core_implementation::SKKNNImputer;
use crate::knn::model::SKKNNImputerModel;
use crate::simple_imputer::SKImputedOutput;

fn assert_send_sync<T: Send + Sync>() {}

fn design(rows: usize, columns: usize) -> Array2<f64> {
    Array2::from_shape_fn((rows, columns), |(row, column)| {
        if (row * 7 + column * 13) % 19 == 0 {
            f64::NAN
        } else {
            ((row * 11 + column * 5) % 37) as f64 / 6.0
        }
    })
}

fn dense(output: SKImputedOutput<f64>) -> Array2<f64> {
    match output {
        SKImputedOutput::Dense(matrix) => matrix,
        SKImputedOutput::Sparse(matrix) => panic!("expected a dense result, found {matrix:?}"),
    }
}

#[test]
fn knn_concurrency_the_model_is_send_and_sync() {
    assert_send_sync::<SKKNNImputerModel<f64>>();
    assert_send_sync::<SKKNNImputerModel<f32>>();
}

#[test]
fn knn_concurrency_one_model_serves_several_threads() {
    let training = design(400, 4);
    let model = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(3)
        .add_indicator(true)
        .build()
        .unwrap()
        .fit(training.view())
        .unwrap();
    let probe = design(240, 4);
    let expected = dense(model.transform(&probe).unwrap());
    let batch = 60;

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
fn knn_concurrency_the_shared_estimator_fits_concurrently() {
    let training = design(120, 3);
    let estimator = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(2)
        .build()
        .unwrap();
    let expected = estimator.fit(training.view()).unwrap().column_means().to_owned();

    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..6)
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
