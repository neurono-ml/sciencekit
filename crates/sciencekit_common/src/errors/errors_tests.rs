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

/// An unsupported operation names both the operation and the reason.
#[test]
fn unsupported_operation_names_the_operation_and_the_reason() {
    let err = SKError::UnsupportedOperation {
        operation: "predict_probabilities",
        reason: "the configured loss is not probabilistic".to_string(),
    };
    match &err {
        SKError::UnsupportedOperation { operation, reason } => {
            assert_eq!(*operation, "predict_probabilities");
            assert!(reason.contains("probabilistic"));
        }
        other => panic!("expected unsupported-operation variant, got {other:?}"),
    }
    let message = format!("{err}");
    assert!(
        message.contains("predict_probabilities") && message.contains("probabilistic"),
        "the rendered message must name the operation and the reason: {message}"
    );
}

/// New missingness names the never-missing feature and the way out.
#[test]
fn unexpected_missingness_names_the_feature() {
    let err = SKError::UnexpectedMissingness { column: 3 };
    let message = format!("{err}");
    assert!(message.contains("3"), "the feature column must be named");
    assert!(message.contains("error_on_new"), "the remedy must be named");
}

/// An empty observation set names the operation that required observations.
#[test]
fn empty_input_names_the_operation() {
    let err = SKError::EmptyInput {
        operation: "accuracy",
    };
    match &err {
        SKError::EmptyInput { operation } => assert_eq!(*operation, "accuracy"),
        other => panic!("expected empty-input variant, got {other:?}"),
    }
    assert!(format!("{err}").contains("accuracy"));
}

/// A non-finite value carries the zero-based position that produced it.
#[test]
fn non_finite_value_carries_its_position() {
    let err = SKError::NonFiniteValue { position: 3 };
    match &err {
        SKError::NonFiniteValue { position } => assert_eq!(*position, 3),
        other => panic!("expected non-finite variant, got {other:?}"),
    }
    assert!(format!("{err}").contains('3'));
}

/// A probability outside the unit interval carries its position.
#[test]
fn probability_out_of_range_carries_the_position() {
    let err = SKError::ProbabilityOutOfRange { position: 4 };
    match &err {
        SKError::ProbabilityOutOfRange { position } => assert_eq!(*position, 4),
        other => panic!("expected probability-out-of-range variant, got {other:?}"),
    }
    assert!(format!("{err}").contains('4'));
}

/// A label outside the resolved space carries the offending value.
#[test]
fn unknown_label_carries_the_label() {
    let err = SKError::UnknownLabel { label: 7 };
    match &err {
        SKError::UnknownLabel { label } => assert_eq!(*label, 7),
        other => panic!("expected unknown-label variant, got {other:?}"),
    }
    assert!(format!("{err}").contains('7'));
}
