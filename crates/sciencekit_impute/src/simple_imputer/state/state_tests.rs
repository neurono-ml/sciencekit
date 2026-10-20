//! Companion tests for the incremental state (spec `simple-imputer`).

use sprs::{CsMat, TriMat};

use super::SKImputationState;
use crate::simple_imputer::strategy::SKImputationStrategy;

/// Build a CSR matrix from triplets for the sparse cases.
fn csr(shape: (usize, usize), entries: &[(usize, usize, f64)]) -> CsMat<f64> {
    let mut triplets = TriMat::new(shape);
    for &(row, column, value) in entries {
        triplets.add_triplet(row, column, value);
    }
    triplets.to_csr()
}

#[test]
fn sparse_mean_counts_the_implicit_zeros() {
    let mut state = SKImputationState::new(1, SKImputationStrategy::Mean);
    state.extend_sparse(
        &csr((3, 1), &[(0, 0, 4.0), (1, 0, 2.0)]).view(),
        SKImputationStrategy::Mean,
        f64::NAN,
    );
    assert_eq!(state.counts, vec![3]);
    assert_eq!(state.statistics(SKImputationStrategy::Mean, None)[0], 2.0);
}

#[test]
fn sparse_median_sketch_weights_the_implicit_zeros() {
    let mut state = SKImputationState::new(1, SKImputationStrategy::Median);
    state.extend_sparse(
        &csr((6, 1), &[(0, 0, 10.0), (1, 0, 12.0)]).view(),
        SKImputationStrategy::Median,
        f64::NAN,
    );
    assert_eq!(state.counts, vec![6], "four structural zeros are observed");
    let median = state.statistics(SKImputationStrategy::Median, None)[0];
    assert!(
        median.abs() < 1e-12,
        "the sorted multiset 0, 0, 0, 0, 10, 12 has median 0, found {median}"
    );
}

#[test]
fn sparse_zero_marker_skips_the_implicit_zeros() {
    let mut state = SKImputationState::new(1, SKImputationStrategy::Mean);
    state.extend_sparse(
        &csr((4, 1), &[(0, 0, 4.0), (1, 0, 8.0)]).view(),
        SKImputationStrategy::Mean,
        0.0,
    );
    assert_eq!(state.counts, vec![2]);
    assert_eq!(state.statistics(SKImputationStrategy::Mean, None)[0], 6.0);
}

#[test]
fn mode_ties_pick_the_smallest_value() {
    let mut state = SKImputationState::new(1, SKImputationStrategy::MostFrequent);
    state.extend_dense(
        &ndarray::array![[5.0], [2.0], [5.0], [2.0]].view(),
        SKImputationStrategy::MostFrequent,
        f64::NAN,
    );
    assert_eq!(
        state.statistics(SKImputationStrategy::MostFrequent, None)[0],
        2.0
    );
}

#[test]
fn empty_state_statistics_are_missing() {
    let state = SKImputationState::<f64>::new(2, SKImputationStrategy::Mean);
    let statistics = state.statistics(SKImputationStrategy::Mean, None);
    assert!(statistics.iter().all(|statistic| statistic.is_nan()));
}
