//! Tests for the central error taxonomy (spec `error-model`).

use std::io;

use super::{SKError, SKError as CentralError};

/// Shape mismatch carries expected and received dimensions.
#[test]
fn shape_mismatch_identifies_dimensions() {
    let err = SKError::shape_mismatch_2d(3, 2, 2, 3);
    match &err {
        SKError::ShapeMismatch { expected, found } => {
            assert_eq!(expected, &vec![3, 2]);
            assert_eq!(found, &vec![2, 3]);
        }
        other => panic!("expected shape mismatch, got {other:?}"),
    }
}

/// Unsupported representation is a distinct variant, not a shape error.
#[test]
fn unsupported_representation_is_distinct_from_shape() {
    let err = SKError::UnsupportedRepresentation {
        representation: "csr",
        suggestion: "convert with to_dense",
    };
    match err {
        SKError::UnsupportedRepresentation {
            representation,
            suggestion,
        } => {
            assert_eq!(representation, "csr");
            assert!(!suggestion.is_empty());
        }
        SKError::ShapeMismatch { .. } => panic!("must not be a shape error"),
        other => panic!("unexpected variant: {other:?}"),
    }
}

/// Non-convergence reports the number of iterations executed.
#[test]
fn non_convergence_reports_effort() {
    let err = SKError::not_converged(250);
    match err {
        SKError::NotConverged { iterations } => assert_eq!(iterations, 250),
        other => panic!("unexpected variant: {other:?}"),
    }
}

/// An algorithm error converts from the central error automatically.
#[test]
fn central_error_propagates_through_algorithm_error() {
    // Simulated per-algorithm error enum with `From<SKError>`.
    #[derive(Debug, thiserror::Error)]
    #[allow(dead_code)]
    enum AlgorithmError {
        #[error(transparent)]
        Central(#[from] CentralError),
        #[error("algorithm-specific failure")]
        Specific,
    }

    let central = SKError::not_converged(10);
    let alg: AlgorithmError = central.into();
    match alg {
        AlgorithmError::Central(SKError::NotConverged { iterations }) => {
            assert_eq!(iterations, 10);
        }
        other => panic!("unexpected: {other:?}"),
    }
}

/// Platform I/O errors convert into the central error with source preserved.
#[test]
fn io_error_becomes_a_library_error_with_source() {
    let io_err = io::Error::new(io::ErrorKind::NotFound, "missing file");
    let central: SKError = io_err.into();
    match &central {
        SKError::Io(source) => {
            assert_eq!(source.kind(), io::ErrorKind::NotFound);
        }
        other => panic!("expected I/O variant, got {other:?}"),
    }
}

/// The error message renders usefully.
#[test]
fn error_display_is_informative() {
    let err = SKError::not_converged(99);
    assert!(format!("{err}").contains("99"));
}
