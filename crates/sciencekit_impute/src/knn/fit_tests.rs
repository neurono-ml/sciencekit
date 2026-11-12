//! Companion tests for the fitted state of the nearest-neighbor imputer
//! (spec `nearest-neighbor-imputation`).

use ndarray::array;
use sciencekit_common::{SKBuilder, SKUnsupervisedFit};

use crate::knn::core_implementation::SKKNNImputer;

fn fitted(features: &ndarray::Array2<f64>, k: usize) -> crate::knn::SKKNNImputerModel<f64> {
    SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(k)
        .build()
        .unwrap()
        .fit(features.view())
        .unwrap()
}

#[test]
fn knn_fit_retains_the_training_design_and_its_mask() {
    let features = array![[1.0_f64, f64::NAN], [2.0, 3.0], [f64::NAN, 4.0]];
    let model = fitted(&features, 2);
    assert_eq!(model.training_features().dim(), features.dim());
    for ((row, column), value) in features.indexed_iter() {
        let retained = model.training_features()[[row, column]];
        assert!(
            (*value - retained).abs() < 1e-12 || (value.is_nan() && retained.is_nan()),
            "cell ({row}, {column}) changed: expected {value}, found {retained}"
        );
    }
    assert_eq!(
        model.missing_mask(),
        array![[false, true], [false, false], [true, false]]
    );
    assert_eq!(model.training_rows(), 3);
    assert_eq!(model.number_of_input_columns(), 2);
}

#[test]
fn knn_fit_records_the_observed_column_means() {
    let features = array![[1.0_f64, f64::NAN], [2.0, 3.0], [f64::NAN, 4.0]];
    let model = fitted(&features, 2);
    assert_eq!(model.column_means().as_slice().unwrap(), &[1.5, 3.5]);
}

#[test]
fn knn_fit_drops_an_all_missing_column_by_default() {
    let features = array![[1.0_f64, 2.0, f64::NAN], [3.0, 4.0, f64::NAN]];
    let model = fitted(&features, 2);
    assert_eq!(model.kept_columns(), &[0, 1]);
    assert_eq!(model.dropped_columns(), &[2]);
    assert_eq!(model.number_of_output_features(), 2);
    assert!(model.column_means()[2].is_nan());
}

#[test]
fn knn_fit_keeps_an_all_missing_column_when_asked() {
    let features = array![[1.0_f64, 2.0, f64::NAN], [3.0, 4.0, f64::NAN]];
    let model = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(2)
        .keep_empty_features(true)
        .build()
        .unwrap()
        .fit(features.view())
        .unwrap();
    assert_eq!(model.kept_columns(), &[0, 1, 2]);
    assert!(model.dropped_columns().is_empty());
    assert_eq!(model.number_of_output_features(), 3);
}

#[test]
fn knn_fit_scalar_marker_normalizes_the_sentinel_for_the_distance_kernel() {
    let features = array![[1.0_f64, -999.0], [2.0, 3.0], [4.0, 5.0]];
    let model = SKKNNImputer::<f64>::build()
        .nearest_neighbors_count(2)
        .missing_values(-999.0)
        .build()
        .unwrap()
        .fit(features.view())
        .unwrap();
    assert!(model.training_features()[[0, 1]].is_nan());
    assert!(model.missing_mask()[[0, 1]]);
    assert_eq!(model.column_means()[1], 4.0);
}

#[test]
fn knn_fit_records_the_donors_of_every_column() {
    let features = array![[1.0_f64, f64::NAN], [2.0, 3.0], [f64::NAN, 4.0]];
    let model = fitted(&features, 2);
    assert_eq!(model.donor_indices(0), &[0, 1]);
    assert_eq!(model.donor_indices(1), &[1, 2]);
}

#[test]
fn knn_fit_leaves_an_all_missing_column_without_donors() {
    let features = array![[1.0_f64, f64::NAN], [3.0, f64::NAN]];
    let model = fitted(&features, 2);
    assert!(model.donor_indices(1).is_empty());
    assert_eq!(model.dropped_columns(), &[1]);
}
