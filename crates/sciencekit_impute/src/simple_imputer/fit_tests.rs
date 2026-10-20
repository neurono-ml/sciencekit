//! Companion tests for the dense `fit` (spec `simple-imputer`).

use ndarray::array;
use sciencekit_common::{SKBackendKind, SKBuilder, SKUnsupervisedFit};
use sprs::{CsMat, TriMat};

use super::core_implementation::SKSimpleImputer;
use super::strategy::SKImputationStrategy;

/// Build a CSR matrix from triplets for the sparse cases.
fn csr(shape: (usize, usize), entries: &[(usize, usize, f64)]) -> CsMat<f64> {
    let mut triplets = TriMat::new(shape);
    for &(row, column, value) in entries {
        triplets.add_triplet(row, column, value);
    }
    triplets.to_csr()
}

#[test]
fn mean_over_the_observed_values() {
    let imputer = SKSimpleImputer::<f64>::build().build().unwrap();
    let model = imputer.fit(&array![[7.0, f64::NAN], [10.0, 1.0]]).unwrap();
    assert_eq!(model.statistics().as_slice().unwrap(), &[8.5, 1.0]);
    assert!(model.dropped_columns().is_empty());
    assert_eq!(model.number_of_input_columns(), 2);
    assert_eq!(model.number_of_output_features(), 2);
}

#[test]
fn median_of_an_odd_observed_count() {
    let imputer = SKSimpleImputer::<f64>::build()
        .strategy(SKImputationStrategy::Median)
        .build()
        .unwrap();
    let model = imputer
        .fit(&array![[1.0], [f64::NAN], [3.0], [5.0]])
        .unwrap();
    assert_eq!(model.statistics().as_slice().unwrap(), &[3.0]);
}

#[test]
fn median_of_an_even_observed_count_averages_the_middle_pair() {
    let imputer = SKSimpleImputer::<f64>::build()
        .strategy(SKImputationStrategy::Median)
        .build()
        .unwrap();
    let model = imputer
        .fit(&array![[1.0], [3.0], [f64::NAN], [9.0], [5.0], [f64::NAN]])
        .unwrap();
    assert_eq!(model.statistics().as_slice().unwrap(), &[4.0]);
}

#[test]
fn most_frequent_tie_breaks_toward_the_smallest() {
    let imputer = SKSimpleImputer::<f64>::build()
        .strategy(SKImputationStrategy::MostFrequent)
        .build()
        .unwrap();
    let model = imputer
        .fit(&array![[2.0, 5.0], [5.0, 1.0], [2.0, 1.0], [f64::NAN, 1.0]])
        .unwrap();
    assert_eq!(model.statistics().as_slice().unwrap(), &[2.0, 1.0]);
}

#[test]
fn constant_fills_every_column_with_the_configured_value() {
    let imputer = SKSimpleImputer::<f64>::build()
        .strategy(SKImputationStrategy::Constant)
        .fill_value(Some(-1.0))
        .build()
        .unwrap();
    let model = imputer
        .fit(&array![[1.0, f64::NAN], [f64::NAN, f64::NAN]])
        .unwrap();
    assert_eq!(model.statistics().as_slice().unwrap(), &[-1.0, -1.0]);
    assert!(
        model.dropped_columns().is_empty(),
        "a constant strategy has no empty columns"
    );
}

#[test]
fn all_missing_column_gets_a_nan_statistic_and_is_reported() {
    let imputer = SKSimpleImputer::<f64>::build().build().unwrap();
    let model = imputer
        .fit(&array![[1.0, f64::NAN], [2.0, f64::NAN]])
        .unwrap();
    let statistics = model.statistics();
    assert!(statistics[1].is_nan());
    assert_eq!(model.dropped_columns(), &[1usize]);
}

#[test]
fn the_model_records_the_carried_backend_kind() {
    let imputer = SKSimpleImputer::<f64>::build()
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap();
    let model = imputer.fit(&array![[1.0]]).unwrap();
    assert_eq!(model.backend(), SKBackendKind::MatrixMultiply);
}

#[test]
fn fit_leaves_the_estimator_reusable_and_deterministic() {
    let imputer = SKSimpleImputer::<f64>::build().build().unwrap();
    let data = array![[1.0, f64::NAN], [3.0, 8.0], [5.0, 10.0]];
    let first = imputer.fit(&data).unwrap();
    let second = imputer.fit(&data).unwrap();
    assert_eq!(
        first.statistics().as_slice().unwrap(),
        second.statistics().as_slice().unwrap()
    );
}

#[test]
fn sparse_mean_counts_the_implicit_zeros_as_observed() {
    let features = csr((3, 2), &[(0, 0, 4.0), (1, 0, 2.0)]);
    let imputer = SKSimpleImputer::<f64>::build().build().unwrap();
    let model = imputer.fit(features.view()).unwrap();
    assert_eq!(
        model.statistics().as_slice().unwrap(),
        &[2.0, 0.0],
        "implicit zeros are observed under a NaN marker"
    );
}

#[test]
fn sparse_explicit_nan_is_excluded_from_the_statistic() {
    let features = csr((3, 1), &[(0, 0, f64::NAN), (1, 0, 6.0)]);
    let imputer = SKSimpleImputer::<f64>::build().build().unwrap();
    let model = imputer.fit(features.view()).unwrap();
    assert_eq!(
        model.statistics().as_slice().unwrap(),
        &[3.0],
        "the observed set is the explicit non-missing entries plus one implicit zero"
    );
}

#[test]
fn sparse_zero_marker_makes_the_implicit_zeros_missing() {
    let features = csr((3, 1), &[(0, 0, 4.0), (1, 0, 2.0)]);
    let imputer = SKSimpleImputer::<f64>::build()
        .missing_values(0.0)
        .build()
        .unwrap();
    let model = imputer.fit(features.view()).unwrap();
    assert_eq!(
        model.statistics().as_slice().unwrap(),
        &[3.0],
        "under a zero marker only the explicit non-zero entries are observed"
    );
}

#[test]
fn sparse_median_includes_the_implicit_zeros() {
    let features = csr((4, 1), &[(0, 0, 10.0)]);
    let imputer = SKSimpleImputer::<f64>::build()
        .strategy(SKImputationStrategy::Median)
        .build()
        .unwrap();
    let model = imputer.fit(features.view()).unwrap();
    assert_eq!(
        model.statistics().as_slice().unwrap(),
        &[0.0],
        "the observed set 10, 0, 0, 0 has median 0"
    );
}

#[test]
fn sparse_mode_counts_the_implicit_zeros() {
    let features = csr((3, 1), &[(0, 0, 5.0)]);
    let imputer = SKSimpleImputer::<f64>::build()
        .strategy(SKImputationStrategy::MostFrequent)
        .build()
        .unwrap();
    let model = imputer.fit(features.view()).unwrap();
    assert_eq!(
        model.statistics().as_slice().unwrap(),
        &[0.0],
        "the observed set 5, 0, 0 has mode 0"
    );
}
