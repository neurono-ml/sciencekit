//! Workspace-level audit of the streaming surface (spec
//! `sparse-streaming-coverage`).
//!
//! Every estimator that offers `fit_streaming` or `partial_fit` is listed in
//! [`STREAMING_COVERAGE`] with the representations it accepts. The per-estimator
//! conformance suites live next to each estimator (their `sparse_streaming_tests`
//! modules); this module is the single place that proves the **set** is complete,
//! so a new streaming estimator that forgets the audit fails here.
//!
//! The dense-only guard is exercised by a local fixture: it shows what a
//! deliberately dense-only streaming estimator must do — reject CSR with
//! `UnsupportedRepresentation` before touching an element, and suggest the
//! alternative path. No shipped estimator is dense-only today.

use sciencekit_common::{SKDataView, SKError};

/// One row of the streaming coverage matrix.
struct StreamingCoverage {
    estimator: &'static str,
    partial_fit: bool,
    dense_streaming: bool,
    sparse_streaming: bool,
    note: &'static str,
}

/// Every `fit_streaming` / `partial_fit` implementation in the workspace.
const STREAMING_COVERAGE: &[StreamingCoverage] = &[
    StreamingCoverage {
        estimator: "SKStandardScaler",
        partial_fit: true,
        dense_streaming: true,
        sparse_streaming: true,
        note: "exact Chan-Golub-LeVeque moments, implicit zeros counted",
    },
    StreamingCoverage {
        estimator: "SKMinMaxScaler",
        partial_fit: true,
        dense_streaming: true,
        sparse_streaming: true,
        note: "exact extrema, implicit zeros included",
    },
    StreamingCoverage {
        estimator: "SKRobustScaler",
        partial_fit: true,
        dense_streaming: true,
        sparse_streaming: true,
        note: "tdigest sketch over stored values",
    },
    StreamingCoverage {
        estimator: "SKPolynomialFeatures",
        partial_fit: false,
        dense_streaming: true,
        sparse_streaming: true,
        note: "expands sparse input directly and emits CSR",
    },
    StreamingCoverage {
        estimator: "SKOneHotEncoder",
        partial_fit: false,
        dense_streaming: true,
        sparse_streaming: false,
        note: "categorical iterator seam: input is categorical, not a numeric block",
    },
    StreamingCoverage {
        estimator: "SKSimpleImputer",
        partial_fit: true,
        dense_streaming: true,
        sparse_streaming: true,
        note: "marker semantics applied over CSR",
    },
    StreamingCoverage {
        estimator: "SKLinearRegression",
        partial_fit: true,
        dense_streaming: true,
        sparse_streaming: true,
        note: "Gram accumulator over stored entries",
    },
    StreamingCoverage {
        estimator: "SKRidge",
        partial_fit: true,
        dense_streaming: true,
        sparse_streaming: true,
        note: "centered Gram accumulator",
    },
    StreamingCoverage {
        estimator: "SKLasso",
        partial_fit: true,
        dense_streaming: true,
        sparse_streaming: true,
        note: "retained design, re-solved at read time",
    },
    StreamingCoverage {
        estimator: "SKElasticNet",
        partial_fit: true,
        dense_streaming: true,
        sparse_streaming: true,
        note: "retained design, re-solved at read time",
    },
    StreamingCoverage {
        estimator: "SKLogisticRegression",
        partial_fit: true,
        dense_streaming: true,
        sparse_streaming: true,
        note: "retained design; dense and CSR share the solver",
    },
    StreamingCoverage {
        estimator: "SKSGDClassifier",
        partial_fit: true,
        dense_streaming: true,
        sparse_streaming: true,
        note: "one epoch per pass, sparse epoch driver",
    },
    StreamingCoverage {
        estimator: "SKSGDRegressor",
        partial_fit: true,
        dense_streaming: true,
        sparse_streaming: true,
        note: "one epoch per pass, sparse epoch driver",
    },
];

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
    assert_eq!(listed, 13, "a streaming estimator was added or removed");
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
}

/// The only streaming estimator without a sparse numeric path is the one
/// documented categorical exception.
#[test]
fn the_only_dense_only_streaming_estimator_is_the_documented_categorical_exception() {
    let dense_only: Vec<&str> = STREAMING_COVERAGE
        .iter()
        .filter(|row| !row.sparse_streaming)
        .map(|row| row.estimator)
        .collect();
    assert_eq!(dense_only, vec!["SKOneHotEncoder"]);
    for row in STREAMING_COVERAGE {
        assert!(row.dense_streaming, "{} must stream dense", row.estimator);
    }
}

/// A deliberately dense-only streaming estimator must reject CSR up front with
/// `UnsupportedRepresentation`, a named representation and a suggestion, before
/// processing any element.
#[test]
fn sparse_streaming_guard_rejects_csr_before_processing() {
    let sparse = sprs::CsMat::new((2, 2), vec![0, 1, 1], vec![0], vec![1.0_f64]);
    let error = sk_dense_only_streaming_fixture(SKDataView::Sparse(sparse.view()))
        .expect_err("a dense-only streaming path must reject CSR");
    match error {
        SKError::UnsupportedRepresentation {
            representation,
            suggestion,
        } => {
            assert_eq!(representation, "csr");
            assert!(
                suggestion.contains("dense"),
                "the suggestion must name the conversion path"
            );
        }
        other => panic!("expected an unsupported-representation error, got {other:?}"),
    }
}

/// The guard accepts the dense representation.
#[test]
fn sparse_streaming_guard_accepts_dense_blocks() {
    let dense = ndarray::array![[1.0_f64, 0.0], [0.0, 2.0]];
    let rows = sk_dense_only_streaming_fixture(SKDataView::Dense(dense.view()))
        .expect("the dense path is supported");
    assert_eq!(rows, 2);
}

/// A stand-in for a dense-only streaming estimator: the representation is
/// matched once, before any element is read.
fn sk_dense_only_streaming_fixture(features: SKDataView<'_, f64>) -> Result<usize, SKError> {
    match features {
        SKDataView::Dense(dense) => Ok(dense.nrows()),
        SKDataView::Sparse(_) => Err(SKError::UnsupportedRepresentation {
            representation: "csr",
            suggestion: "convert the block to dense with to_dense, or use a sparse-aware estimator",
        }),
        _ => Err(SKError::UnsupportedRepresentation {
            representation: "unknown",
            suggestion: "convert the input to a supported dense or CSR view",
        }),
    }
}
