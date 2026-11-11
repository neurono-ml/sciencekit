//! Companion tests for the streaming coverage matrix.

use super::matrix::{PARTIAL_FIT_METHODS, STREAMING_COVERAGE, STREAMING_ESTIMATORS};

/// The matrix lists every streaming estimator exactly once.
#[test]
fn every_streaming_implementation_is_in_the_coverage_matrix() {
    let mut names: Vec<&str> = STREAMING_COVERAGE.iter().map(|row| row.estimator).collect();
    let listed = names.len();
    names.sort_unstable();
    names.dedup();
    assert_eq!(
        names.len(),
        listed,
        "the matrix must not repeat an estimator"
    );
    assert_eq!(
        listed, STREAMING_ESTIMATORS,
        "a streaming estimator was added or removed"
    );
    for row in STREAMING_COVERAGE {
        assert!(
            !row.note.is_empty(),
            "{} needs a coverage note justifying its representation support",
            row.estimator
        );
    }
    for expected in [
        "SKStandardScaler",
        "SKMinMaxScaler",
        "SKRobustScaler",
        "SKPolynomialFeatures",
        "SKOneHotEncoder",
        "SKSimpleImputer",
        "SKLinearRegression",
        "SKRidge",
        "SKLasso",
        "SKElasticNet",
        "SKLogisticRegression",
        "SKSGDClassifier",
        "SKSGDRegressor",
        "SKIncrementalPCA",
    ] {
        assert!(
            names.contains(&expected),
            "{expected} is missing from the streaming coverage matrix"
        );
    }
}

/// `partial_fit` is offered by every estimator except the two stateless
/// transformers that only learn at fit time.
#[test]
fn partial_fit_coverage_matches_the_documented_exceptions() {
    let without_partial_fit: Vec<&str> = STREAMING_COVERAGE
        .iter()
        .filter(|row| !row.partial_fit)
        .map(|row| row.estimator)
        .collect();
    assert_eq!(
        without_partial_fit,
        vec!["SKPolynomialFeatures", "SKOneHotEncoder"]
    );
    let with_partial_fit = STREAMING_COVERAGE
        .iter()
        .filter(|row| row.partial_fit)
        .count();
    assert_eq!(with_partial_fit, PARTIAL_FIT_METHODS);
}

/// The only streaming seam without a numeric sparse path is the documented
/// categorical exception.
#[test]
fn the_only_dense_only_streaming_estimator_is_the_documented_categorical_exception() {
    let not_sparse: Vec<&str> = STREAMING_COVERAGE
        .iter()
        .filter(|row| !row.sparse_streaming)
        .map(|row| row.estimator)
        .collect();
    assert_eq!(not_sparse, vec!["SKOneHotEncoder"]);
    for row in STREAMING_COVERAGE
        .iter()
        .filter(|row| row.numeric_streaming)
    {
        assert!(
            row.sparse_streaming,
            "{} streams numeric blocks, so it must accept CSR",
            row.estimator
        );
    }
}
