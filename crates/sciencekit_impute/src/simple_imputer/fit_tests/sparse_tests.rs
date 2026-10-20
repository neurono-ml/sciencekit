//! Companion tests for the sparse batch fit (spec `simple-imputer`), where the
//! structural zeros join the observed set under the `NaN` marker.

use sciencekit_common::{SKBuilder, SKUnsupervisedFit};
use sprs::{CsMat, TriMat};

use crate::simple_imputer::core_implementation::SKSimpleImputer;
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
