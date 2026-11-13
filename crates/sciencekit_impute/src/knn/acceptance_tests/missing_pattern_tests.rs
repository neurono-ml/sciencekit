//! Every missing pattern acceptance for the nearest-neighbor imputer.

use ndarray::Array2;
use sciencekit_common::{SKBuilder, SKFeatureTransformer, SKUnsupervisedFit};

use super::fixtures::{dense, linear_design};
use crate::knn::core_implementation::SKKNNImputer;

const COLUMNS: usize = 3;

#[test]
fn acceptance_knn_imputer_every_missing_pattern_is_filled() {
    let training = linear_design(30, COLUMNS);
    let model = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(3)
        .build()
        .unwrap()
        .fit(training.view())
        .unwrap();

    // One receiver per subset of missing columns, eight patterns in total.
    let mut probe = Array2::zeros((1 << COLUMNS, COLUMNS));
    let mut missing = Array2::from_elem((1 << COLUMNS, COLUMNS), false);
    for pattern in 0..(1 << COLUMNS) {
        for column in 0..COLUMNS {
            let is_missing = pattern & (1 << column) != 0;
            probe[[pattern, column]] = 12.0 * (column + 1) as f64;
            if is_missing {
                probe[[pattern, column]] = f64::NAN;
                missing[[pattern, column]] = true;
            }
        }
    }

    let output = dense(model.transform(&probe).unwrap());
    assert_eq!(output.dim(), (1 << COLUMNS, COLUMNS));
    for row in 0..output.nrows() {
        for column in 0..COLUMNS {
            if missing[[row, column]] {
                assert!(
                    output[[row, column]].is_finite(),
                    "pattern {row}, column {column} was not filled"
                );
            } else {
                assert_eq!(output[[row, column]], probe[[row, column]]);
            }
        }
    }
}

#[test]
fn acceptance_knn_imputer_a_row_missing_every_feature_uses_the_column_means() {
    let training = linear_design(10, COLUMNS);
    let model = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(2)
        .build()
        .unwrap()
        .fit(training.view())
        .unwrap();
    let output = dense(
        model
            .transform(&Array2::from_elem((1, COLUMNS), f64::NAN))
            .unwrap(),
    );
    for (column, mean) in model.column_means().iter().enumerate() {
        assert!((output[[0, column]] - mean).abs() < 1e-12);
    }
}
