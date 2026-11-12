//! TDD tests for seeded and stratified resource subsampling
//! (spec `halving-grid-search`, requirement `Resource subsampling`).

use std::collections::HashSet;

use ndarray::{Array1, Array2};
use sciencekit_common::{SKBuilder, SKDataView, SKTargetView};
use sprs::TriMat;

use crate::split::kfold::SKKFold;
use crate::split::stratified::SKStratifiedKFold;
use crate::{SKSplitter, sk_gather_rows};

use super::subsample::{sk_gather_round_subsample, sk_select_subsample_indices};

#[test]
fn halving_subsample_draws_exact_resource_count_and_is_reproducible_per_seed_and_round() {
    let targets = Array1::from_iter((0..100).map(|row| row as f64));
    let target_view: SKTargetView<'_, f64> = targets.view().try_into().unwrap();

    let first = sk_select_subsample_indices(100, 30, 42, 0, false, target_view, None).unwrap();
    let repeated = sk_select_subsample_indices(100, 30, 42, 0, false, target_view, None).unwrap();
    let next_round = sk_select_subsample_indices(100, 30, 42, 1, false, target_view, None).unwrap();
    let other_seed = sk_select_subsample_indices(100, 30, 99, 0, false, target_view, None).unwrap();

    assert_eq!(first.len(), 30);
    assert_eq!(first.iter().copied().collect::<HashSet<_>>().len(), 30);
    assert!(first.iter().all(|&index| index < 100));
    assert_eq!(first, repeated);
    assert_ne!(first, next_round);
    assert_ne!(first, other_seed);
}

#[test]
fn halving_subsample_stratified_preserves_class_proportions_for_integer_and_nominal_targets() {
    // 70 rows of class 0 and 30 rows of class 1 -> a 30-row subsample must contain 21 and 9.
    let integer_labels = Array1::from_iter((0..100).map(|row| i64::from(row >= 70)));
    let integer_view: SKTargetView<'_, f64> = integer_labels.view().try_into().unwrap();

    let indices = sk_select_subsample_indices(100, 30, 7, 0, true, integer_view, None).unwrap();
    assert_eq!(indices.len(), 30);
    let class_zero_count = indices
        .iter()
        .filter(|&&row| integer_labels[row] == 0)
        .count();
    let class_one_count = indices
        .iter()
        .filter(|&&row| integer_labels[row] == 1)
        .count();
    assert_eq!((class_zero_count, class_one_count), (21, 9));

    let nominal_labels: Vec<&str> = (0..100)
        .map(|row| if row < 70 { "alpha" } else { "beta" })
        .collect();
    let nominal_view = SKTargetView::<f64>::Nominal(nominal_labels.as_slice());
    let nominal_indices =
        sk_select_subsample_indices(100, 30, 7, 0, true, nominal_view, None).unwrap();
    let alpha_count = nominal_indices
        .iter()
        .filter(|&&row| nominal_labels[row] == "alpha")
        .count();
    assert_eq!(alpha_count, 21);
}

#[test]
fn halving_subsample_preserves_sparse_csr_representation_and_groups() {
    let mut triplets = TriMat::new((30, 4));
    (0..30).for_each(|row| triplets.add_triplet(row, row % 4, (row + 1) as f64));
    let sparse_features = triplets.to_csr();
    let targets = Array1::from_iter((0..30).map(|row| (row % 2) as i64));
    let groups = Array1::from_iter((0..30).map(|row| (row / 10) as i64));

    let subsample = sk_gather_round_subsample(
        sparse_features.view().try_into().unwrap(),
        targets.view().try_into().unwrap(),
        Some(groups.view()),
        15,
        11,
        0,
        false,
    )
    .unwrap();

    assert_eq!(subsample.row_indices.len(), 15);
    assert!(matches!(subsample.features.view(), SKDataView::Sparse(view) if view.rows() == 15));
    let sub_groups = subsample.groups.expect("groups were supplied");
    assert_eq!(sub_groups.len(), 15);
    // Each of the 3 equal-sized groups of 10 rows receives 5 rows in the 15-row subsample.
    assert_eq!(sub_groups.iter().filter(|&&g| g == 0).count(), 5);
    assert_eq!(sub_groups.iter().filter(|&&g| g == 1).count(), 5);
    assert_eq!(sub_groups.iter().filter(|&&g| g == 2).count(), 5);

    let (expected_features, _) = sk_gather_rows(
        sparse_features.view().try_into().unwrap(),
        Some(targets.view().try_into().unwrap()),
        &subsample.row_indices,
    )
    .unwrap();
    let SKDataView::Sparse(actual_csr) = subsample.features.view() else {
        panic!("expected sparse CSR");
    };
    let SKDataView::Sparse(expected_csr) = expected_features.view() else {
        panic!("expected sparse CSR");
    };
    assert_eq!(actual_csr.data(), expected_csr.data());
}

#[test]
fn halving_subsample_full_budget_returns_all_rows_and_stratified_splitter_signals_requirement() {
    let features = Array2::from_shape_fn((12, 2), |(row, col)| (row + col) as f64);
    let targets = Array1::from_iter((0..12).map(|row| i64::from(row >= 6)));

    let subsample = sk_gather_round_subsample(
        features.view().try_into().unwrap(),
        targets.view().try_into().unwrap(),
        None,
        12,
        0,
        2,
        true,
    )
    .unwrap();
    assert_eq!(subsample.row_indices, (0..12).collect::<Vec<_>>());

    let kfold = SKKFold::build().number_of_splits(3).build().unwrap();
    let stratified = SKStratifiedKFold::build()
        .number_of_splits(3)
        .build()
        .unwrap();
    assert!(!<SKKFold as SKSplitter<f64>>::requires_stratification(
        &kfold
    ));
    assert!(<SKStratifiedKFold as SKSplitter<f64>>::requires_stratification(&stratified));
}
