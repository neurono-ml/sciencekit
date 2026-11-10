//! Companion tests for the dense transform of the nearest-neighbor imputer
//! (spec `nearest-neighbor-imputation`).

use ndarray::{Array2, array};
use sciencekit_common::{SKBuilder, SKFeatureTransformer, SKUnsupervisedFit};

use crate::knn::core_implementation::SKKNNImputer;
use crate::simple_imputer::SKImputedOutput;

fn dense(output: SKImputedOutput<f64>) -> Array2<f64> {
    match output {
        SKImputedOutput::Dense(matrix) => matrix,
        SKImputedOutput::Sparse(matrix) => panic!("expected a dense result, found {matrix:?}"),
    }
}

fn fitted(features: &Array2<f64>, k: usize) -> crate::knn::SKKNNImputerModel<f64> {
    SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(k)
        .build()
        .unwrap()
        .fit(features.view())
        .unwrap()
}

#[test]
fn knn_transform_imputes_a_known_matrix() {
    // Column 1 has donors 2, 4 and 6 at distances 0, √2 and √8 from the
    // receiver [0, NaN]; the two nearest average to 3.
    let training = array![[0.0_f64, 2.0], [1.0, 4.0], [2.0, 6.0]];
    let model = fitted(&training, 2);
    let output = dense(model.transform(&array![[0.0_f64, f64::NAN]]).unwrap());
    assert_eq!(output, array![[0.0_f64, 3.0]]);
}

#[test]
fn knn_transform_preserves_observed_cells_bit_identically() {
    let training = array![[0.0_f64, 2.0], [1.0, 4.0], [2.0, 6.0]];
    let model = fitted(&training, 1);
    let probe = array![[0.25_f64, 7.0], [0.0, f64::NAN]];
    let output = dense(model.transform(&probe).unwrap());
    assert_eq!(output[[0, 0]], 0.25);
    assert_eq!(output[[0, 1]], 7.0);
    assert_eq!(output[[1, 0]], 0.0);
}

#[test]
fn knn_transform_falls_back_to_the_column_mean_without_a_comparable_donor() {
    let training = array![[0.0_f64, 2.0], [1.0, 4.0], [2.0, 6.0]];
    let model = fitted(&training, 2);
    let output = dense(model.transform(&array![[f64::NAN, f64::NAN]]).unwrap());
    assert_eq!(output, array![[1.0_f64, 4.0]]);
}

#[test]
fn knn_transform_drops_an_all_missing_column() {
    let training = array![[1.0_f64, f64::NAN], [2.0, f64::NAN]];
    let model = fitted(&training, 2);
    let output = dense(model.transform(&array![[f64::NAN, f64::NAN]]).unwrap());
    assert_eq!(output.dim(), (1, 1));
    assert_eq!(output[[0, 0]], 1.5);
}

#[test]
fn knn_transform_keeps_an_all_missing_column_at_zero() {
    let training = array![[1.0_f64, f64::NAN], [2.0, f64::NAN]];
    let model = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(2)
        .keep_empty_features(true)
        .build()
        .unwrap()
        .fit(training.view())
        .unwrap();
    let output = dense(model.transform(&array![[f64::NAN, f64::NAN]]).unwrap());
    assert_eq!(output, array![[1.5_f64, 0.0]]);
}

#[test]
fn knn_transform_respects_a_scalar_marker() {
    let training = array![[1.0_f64, -999.0], [2.0, 3.0], [3.0, 5.0]];
    let model = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(2)
        .missing_values(-999.0)
        .build()
        .unwrap()
        .fit(training.view())
        .unwrap();
    let output = dense(model.transform(&array![[4.0_f64, -999.0]]).unwrap());
    assert_eq!(output, array![[4.0_f64, 4.0]]);
}

#[test]
fn knn_transform_works_with_a_single_donor_and_one_receiver() {
    let training = array![[1.0_f64, 2.0]];
    let model = fitted(&training, 1);
    let output = dense(model.transform(&array![[f64::NAN, f64::NAN]]).unwrap());
    assert_eq!(output, array![[1.0_f64, 2.0]]);
}

#[test]
fn knn_transform_does_not_impute_observed_rows() {
    let training = array![[0.0_f64, 2.0], [1.0, 4.0], [2.0, 6.0]];
    let model = fitted(&training, 2);
    let probe = array![[5.0_f64, 8.0], [6.0, 9.0]];
    let output = dense(model.transform(&probe).unwrap());
    assert_eq!(output, probe);
}

#[test]
fn knn_transform_parallel_equals_sequential() {
    let training = Array2::from_shape_fn((1_500, 4), |(row, column)| {
        ((row * 7 + column * 13) % 31) as f64 / 4.0
    });
    let model = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(3)
        .build()
        .unwrap()
        .fit(training.view())
        .unwrap();
    let probe = Array2::from_shape_fn((1_200, 4), |(row, column)| {
        if (row * 5 + column * 3) % 17 == 0 {
            f64::NAN
        } else {
            ((row * 11 + column * 2) % 29) as f64 / 5.0
        }
    });

    assert!(
        model.distance_block_rows() * training.nrows() * size_of::<f64>()
            <= crate::knn::distance_block::SK_DISTANCE_BLOCK_BYTES,
        "the block must respect the byte budget"
    );
    assert!(
        model.distance_block_rows() < probe.nrows(),
        "the plan must split a large receiver set into several blocks"
    );

    let parallel = model
        .transform_dense_with_parallelism(&probe.view(), true)
        .unwrap();
    let sequential = model
        .transform_dense_with_parallelism(&probe.view(), false)
        .unwrap();
    assert_eq!(parallel, sequential);

    for row in 0..probe.nrows() {
        let single = probe.slice(ndarray::s![row..row + 1, ..]);
        let expected = model
            .transform_dense_with_parallelism(&single, false)
            .unwrap();
        for (found, wanted) in parallel.row(row).iter().zip(expected.row(0)) {
            assert!(
                (found - wanted).abs() < 1e-12 || (found.is_nan() && wanted.is_nan()),
                "row {row} differs from the row-by-row result"
            );
        }
    }
}

/// A column that was all-missing at fit and retained by `keep_empty_features`
/// is emitted as zeros — even for a receiver row that observes it.
#[test]
fn knn_transform_keeps_a_retained_empty_column_at_zero() {
    let training = ndarray::array![[1.0_f64, f64::NAN], [2.0, f64::NAN]];
    let model = crate::knn::SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(1)
        .keep_empty_features(true)
        .build()
        .unwrap()
        .fit(training.view())
        .unwrap();
    let receiver = ndarray::array![[f64::NAN, 7.0_f64]];
    let output = dense(model.transform(receiver.view()).unwrap());
    assert!((output[[0, 0]] - 1.5).abs() < 1e-12);
    assert_eq!(output[[0, 1]], 0.0);
}
