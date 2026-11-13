//! Companion tests for per-column donor selection
//! (spec `nearest-neighbor-imputation`).

use ndarray::array;

use crate::knn::donors::{sk_column_donors, sk_select_nearest_donors};

#[test]
fn knn_donors_a_row_missing_the_column_is_not_a_donor_for_it() {
    let mask = array![[false, true], [false, false], [true, false]];
    let donors = sk_column_donors(&mask.view());
    assert_eq!(donors[0], vec![0, 1]);
    assert_eq!(donors[1], vec![1, 2]);
}

#[test]
fn knn_donors_an_all_missing_column_has_no_potential_donor() {
    let mask = array![[false, true], [false, true]];
    let donors = sk_column_donors(&mask.view());
    assert!(donors[1].is_empty());
}

#[test]
fn knn_donors_selects_the_k_smallest_distances() {
    let distances = array![4.0_f64, 1.0, 3.0, 2.0];
    let selected = sk_select_nearest_donors(&distances.view(), &[0, 1, 2, 3], 2);
    assert_eq!(selected, vec![1, 3]);
}

#[test]
fn knn_donors_skips_nan_distances() {
    let distances = array![f64::NAN, 1.0, f64::NAN, 2.0];
    let selected = sk_select_nearest_donors(&distances.view(), &[0, 1, 2, 3], 3);
    assert_eq!(selected, vec![1, 3]);
}

#[test]
fn knn_donors_with_no_finite_distance_selects_nothing() {
    let distances = array![f64::NAN, f64::NAN];
    assert!(sk_select_nearest_donors(&distances.view(), &[0, 1], 2).is_empty());
}

#[test]
fn knn_donors_returns_every_donor_when_fewer_than_k_exist() {
    let distances = array![2.0_f64, 1.0, 3.0];
    let selected = sk_select_nearest_donors(&distances.view(), &[0, 1, 2], 10);
    assert_eq!(selected, vec![1, 0, 2]);
}

#[test]
fn knn_donors_ties_break_by_the_smallest_training_index() {
    let distances = array![2.0_f64, 2.0, 2.0, 1.0];
    let selected = sk_select_nearest_donors(&distances.view(), &[3, 2, 1, 0], 3);
    assert_eq!(selected, vec![3, 0, 1]);
}

#[test]
fn knn_donors_only_considers_the_given_column_donors() {
    let distances = array![0.0_f64, 5.0, 3.0];
    let selected = sk_select_nearest_donors(&distances.view(), &[1, 2], 2);
    assert_eq!(selected, vec![2, 1]);
}
