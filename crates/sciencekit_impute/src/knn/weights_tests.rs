//! Companion tests for donor averaging and the column-mean fallback
//! (spec `nearest-neighbor-imputation`).

use ndarray::array;

use crate::knn::donors::sk_impute_cell;
use crate::knn::weights::SKImputationWeights;

fn close(left: f64, right: f64) {
    assert!(
        (left - right).abs() < 1e-12,
        "expected {right}, found {left}"
    );
}

#[test]
fn knn_average_uniform_mean_of_the_selected_donors() {
    let distances = array![1.0_f64, 2.0, 3.0];
    let values = array![2.0_f64, 4.0, 6.0];
    let imputed = sk_impute_cell(
        &distances.view(),
        &[0, 1, 2],
        &values.view(),
        4.0,
        3,
        SKImputationWeights::Uniform,
    );
    close(imputed, 4.0);
}

#[test]
fn knn_average_uniform_uses_only_the_nearest_donors() {
    let distances = array![1.0_f64, 5.0, 6.0];
    let values = array![2.0_f64, 100.0, 200.0];
    let imputed = sk_impute_cell(
        &distances.view(),
        &[0, 1, 2],
        &values.view(),
        100.0,
        1,
        SKImputationWeights::Uniform,
    );
    close(imputed, 2.0);
}

#[test]
fn knn_average_distance_weighting_matches_the_hand_computed_value() {
    let distances = array![1.0_f64, 2.0, 4.0];
    let values = array![2.0_f64, 4.0, 6.0];
    // (1·2 + 0.5·4 + 0.25·6) / (1 + 0.5 + 0.25) = 5.5 / 1.75
    let imputed = sk_impute_cell(
        &distances.view(),
        &[0, 1, 2],
        &values.view(),
        4.0,
        3,
        SKImputationWeights::Distance,
    );
    close(imputed, 5.5 / 1.75);
}

#[test]
fn knn_average_zero_distance_donors_take_the_whole_weight() {
    let distances = array![0.0_f64, 0.0, 5.0];
    let values = array![10.0_f64, 20.0, 30.0];
    let imputed = sk_impute_cell(
        &distances.view(),
        &[0, 1, 2],
        &values.view(),
        20.0,
        3,
        SKImputationWeights::Distance,
    );
    close(imputed, 15.0);
}

#[test]
fn knn_average_falls_back_to_the_column_mean_when_no_distance_is_finite() {
    let distances = array![f64::NAN, f64::NAN];
    let values = array![3.0_f64, 5.0];
    let imputed = sk_impute_cell(
        &distances.view(),
        &[0, 1],
        &values.view(),
        4.0,
        2,
        SKImputationWeights::Distance,
    );
    close(imputed, 4.0);
}

#[test]
fn knn_average_an_empty_column_fills_with_zero() {
    let distances = array![1.0_f64, 2.0];
    let values = array![f64::NAN, f64::NAN];
    let imputed = sk_impute_cell(
        &distances.view(),
        &[],
        &values.view(),
        f64::NAN,
        2,
        SKImputationWeights::Uniform,
    );
    assert_eq!(imputed, 0.0);
}

#[test]
fn knn_average_nan_distance_donors_are_never_averaged_in() {
    // The finite donor wins even though the NaN-distance donor is closer in
    // the candidate list.
    let distances = array![f64::NAN, 2.0];
    let values = array![999.0_f64, 4.0];
    let imputed = sk_impute_cell(
        &distances.view(),
        &[0, 1],
        &values.view(),
        500.0,
        2,
        SKImputationWeights::Uniform,
    );
    close(imputed, 4.0);
}
