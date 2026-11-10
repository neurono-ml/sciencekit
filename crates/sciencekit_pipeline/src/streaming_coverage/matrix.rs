//! The coverage matrix: one row per streaming estimator.

/// One row of the streaming coverage matrix.
pub(super) struct StreamingCoverage {
    /// The estimator type name.
    pub(super) estimator: &'static str,
    /// Whether the model offers `partial_fit`.
    pub(super) partial_fit: bool,
    /// Whether the streaming seam takes a numeric feature block (as opposed to
    /// the categorical iterator seam of the one-hot encoder).
    pub(super) numeric_streaming: bool,
    /// Whether the streaming seam accepts a CSR block.
    pub(super) sparse_streaming: bool,
    /// Why the row has this representation support.
    pub(super) note: &'static str,
}

/// Every `fit_streaming` / `partial_fit` implementation in the workspace.
pub(super) const STREAMING_COVERAGE: &[StreamingCoverage] = &[
    StreamingCoverage {
        estimator: "SKStandardScaler",
        partial_fit: true,
        numeric_streaming: true,
        sparse_streaming: true,
        note: "exact Chan-Golub-LeVeque moments, implicit zeros counted",
    },
    StreamingCoverage {
        estimator: "SKMinMaxScaler",
        partial_fit: true,
        numeric_streaming: true,
        sparse_streaming: true,
        note: "exact extrema, implicit zeros included",
    },
    StreamingCoverage {
        estimator: "SKRobustScaler",
        partial_fit: true,
        numeric_streaming: true,
        sparse_streaming: true,
        note: "tdigest sketch over stored values",
    },
    StreamingCoverage {
        estimator: "SKPolynomialFeatures",
        partial_fit: false,
        numeric_streaming: true,
        sparse_streaming: true,
        note: "expands sparse input directly and emits CSR",
    },
    StreamingCoverage {
        estimator: "SKOneHotEncoder",
        partial_fit: false,
        numeric_streaming: false,
        sparse_streaming: false,
        note: "categorical iterator seam: input is categorical, not a numeric block",
    },
    StreamingCoverage {
        estimator: "SKSimpleImputer",
        partial_fit: true,
        numeric_streaming: true,
        sparse_streaming: true,
        note: "marker semantics applied over CSR",
    },
    StreamingCoverage {
        estimator: "SKLinearRegression",
        partial_fit: true,
        numeric_streaming: true,
        sparse_streaming: true,
        note: "Gram accumulator over stored entries",
    },
    StreamingCoverage {
        estimator: "SKRidge",
        partial_fit: true,
        numeric_streaming: true,
        sparse_streaming: true,
        note: "centered Gram accumulator",
    },
    StreamingCoverage {
        estimator: "SKLasso",
        partial_fit: true,
        numeric_streaming: true,
        sparse_streaming: true,
        note: "retained design, re-solved at read time",
    },
    StreamingCoverage {
        estimator: "SKElasticNet",
        partial_fit: true,
        numeric_streaming: true,
        sparse_streaming: true,
        note: "retained design, re-solved at read time",
    },
    StreamingCoverage {
        estimator: "SKLogisticRegression",
        partial_fit: true,
        numeric_streaming: true,
        sparse_streaming: true,
        note: "retained design; dense and CSR share the solver",
    },
    StreamingCoverage {
        estimator: "SKSGDClassifier",
        partial_fit: true,
        numeric_streaming: true,
        sparse_streaming: true,
        note: "one epoch per pass, sparse epoch driver",
    },
    StreamingCoverage {
        estimator: "SKSGDRegressor",
        partial_fit: true,
        numeric_streaming: true,
        sparse_streaming: true,
        note: "one epoch per pass, sparse epoch driver",
    },
];

/// The number of streaming estimators the audit expects.
pub(super) const STREAMING_ESTIMATORS: usize = 13;

/// The number of `partial_fit` methods the audit expects.
pub(super) const PARTIAL_FIT_METHODS: usize = 11;

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
