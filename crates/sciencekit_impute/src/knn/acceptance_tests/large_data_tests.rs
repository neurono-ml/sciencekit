//! Large-data acceptance for the nearest-neighbor imputer.

use ndarray::Array2;
use sciencekit_common::{SKBuilder, SKFeatureTransformer, SKUnsupervisedFit};

use super::fixtures::{dense, linear_design};
use crate::knn::core_implementation::SKKNNImputer;

const TRAINING_ROWS: usize = 10_000;
const COLUMNS: usize = 4;
const RECEIVERS: usize = 200;

#[test]
fn acceptance_knn_imputer_fits_ten_thousand_rows_and_transforms() {
    let training = linear_design(TRAINING_ROWS, COLUMNS);
    let model = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(5)
        .build()
        .unwrap()
        .fit(training.view())
        .unwrap();
    assert_eq!(model.training_rows(), TRAINING_ROWS);
    assert!(
        model.distance_block_rows() < RECEIVERS,
        "a ten-thousand-row training design must split the receivers into blocks"
    );

    let mut probe = Array2::from_shape_fn((RECEIVERS, COLUMNS), |(row, column)| {
        (TRAINING_ROWS - RECEIVERS + row) as f64 * 0.5 * (column + 1) as f64
    });
    for row in 0..RECEIVERS {
        probe[[row, 1]] = f64::NAN;
    }
    let output = dense(model.transform(&probe).unwrap());
    assert_eq!(output.dim(), (RECEIVERS, COLUMNS));
    assert!(output.iter().all(|value| value.is_finite()));

    let first_row = probe.slice(ndarray::s![0..1, ..]).to_owned();
    let single = dense(model.transform(&first_row).unwrap());
    for (found, expected) in output.row(0).iter().zip(single.row(0)) {
        assert!((found - expected).abs() < 1e-9);
    }
}

#[test]
fn acceptance_knn_imputer_large_transform_is_deterministic() {
    let training = linear_design(TRAINING_ROWS, COLUMNS);
    let model = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(3)
        .weights(crate::knn::SKImputationWeights::Distance)
        .build()
        .unwrap()
        .fit(training.view())
        .unwrap();
    let mut probe = linear_design(RECEIVERS, COLUMNS);
    for row in (0..RECEIVERS).step_by(3) {
        probe[[row, 2]] = f64::NAN;
    }
    let first = dense(model.transform(&probe).unwrap());
    for _ in 0..3 {
        assert_eq!(dense(model.transform(&probe).unwrap()), first);
    }
}
