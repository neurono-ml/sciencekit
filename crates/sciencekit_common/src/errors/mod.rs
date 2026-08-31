//! Central error taxonomy shared across the whole library (spec `error-model`).
//!
//! `SKError` names precisely the common failures every algorithm can produce.
//! Algorithm-specific error enums convert from it automatically via
//! [`From<SKError>`], keeping common errors identical across algorithms.

use std::io;
use thiserror::Error;

/// The central error enum covering the common failure modes of the library.
#[derive(Debug, Error)]
#[non_exhaustive]
pub enum SKError {
    /// The operation received data whose dimensions are incompatible with it.
    #[error("shape mismatch: expected {expected:?}, found {found:?}")]
    ShapeMismatch {
        /// The expected shape.
        expected: Vec<usize>,
        /// The shape actually received.
        found: Vec<usize>,
    },

    /// An algorithm supporting only part of the representations received an
    /// unsupported one. Distinct from a shape error.
    #[error(
        "data representation `{representation}` is not supported by this algorithm; {suggestion}"
    )]
    UnsupportedRepresentation {
        /// The representation that was rejected.
        representation: &'static str,
        /// A conversion path the consumer can take.
        suggestion: &'static str,
    },

    /// An identifiable invalid hyperparameter value.
    #[error("invalid hyperparameter `{name}`: {reason}")]
    InvalidHyperparameter {
        /// The name of the offending hyperparameter.
        name: &'static str,
        /// Why the value is invalid.
        reason: String,
    },

    /// An explicitly requested execution mode incompatible with the algorithm's
    /// declared access pattern.
    #[error("execution mode `{mode}` is incompatible with the declared access pattern `{pattern}`")]
    ExecutionModeIncompatible {
        /// The requested mode.
        mode: &'static str,
        /// The declared access pattern.
        pattern: &'static str,
    },

    /// An iterative process exhausted its iterations without converging.
    #[error("failed to converge after {iterations} iterations")]
    NotConverged {
        /// The number of iterations executed.
        iterations: usize,
    },

    /// A platform I/O error, with the original error preserved as source.
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),

    /// A conversion failure at the data boundary.
    #[error("conversion failure: {0}")]
    Conversion(String),
}

/// Convenience constructors for the most common variants.
impl SKError {
    /// Build a shape-mismatch error for a 2-D input (rows, columns).
    pub fn shape_mismatch_2d(
        expected_rows: usize,
        expected_cols: usize,
        found_rows: usize,
        found_cols: usize,
    ) -> Self {
        SKError::ShapeMismatch {
            expected: vec![expected_rows, expected_cols],
            found: vec![found_rows, found_cols],
        }
    }

    /// Build a not-converged error reporting the effort spent.
    pub fn not_converged(iterations: usize) -> Self {
        SKError::NotConverged { iterations }
    }
}

#[cfg(test)]
mod errors_tests;
