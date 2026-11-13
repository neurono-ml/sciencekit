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
    StreamingCoverage {
        estimator: "SKIncrementalPCA",
        partial_fit: true,
        numeric_streaming: true,
        sparse_streaming: true,
        note: "Chan-Golub-LeVeque scatter merge with Neumaier-compensated CSR Gram accumulation",
    },
    StreamingCoverage {
        estimator: "SKMiniBatchKMeans",
        partial_fit: true,
        numeric_streaming: true,
        sparse_streaming: true,
        note: "per-center learning rates over dense and CSR batches without densifying",
    },
];

/// The number of streaming estimators the audit expects.
pub(super) const STREAMING_ESTIMATORS: usize = 15;

/// The number of `partial_fit` methods the audit expects.
pub(super) const PARTIAL_FIT_METHODS: usize = 13;
