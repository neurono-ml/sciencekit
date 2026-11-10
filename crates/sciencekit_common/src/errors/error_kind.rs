//! The central error enum and its convenience constructors.

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

    /// An operation requested from an otherwise valid configuration, which that
    /// configuration cannot perform.
    ///
    /// Distinct from [`SKError::UnsupportedRepresentation`], which reports an
    /// input representation the algorithm cannot consume, and from
    /// [`SKError::InvalidHyperparameter`], which reports a value rejected when
    /// the configuration was built.
    #[error("operation `{operation}` is not supported: {reason}")]
    UnsupportedOperation {
        /// The name of the requested operation.
        operation: &'static str,
        /// Why the operation is unavailable.
        reason: String,
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

    /// A streaming batch hint whose double-buffered footprint exceeds available
    /// memory. Raised at resolution, before any data is processed.
    #[error(
        "streaming batch buffer overflow: {required_bytes} bytes needed (2 × {batch_bytes} per batch) exceeds {available_bytes} bytes available"
    )]
    BatchBufferOversize {
        /// The byte size of one resident batch.
        batch_bytes: u64,
        /// The total resident bytes required by double buffering (2 × `batch_bytes`).
        required_bytes: u64,
        /// The available memory in bytes.
        available_bytes: u64,
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

    /// Missingness met at transform in a feature that was complete at fit
    /// time, while `error_on_new` is enabled.
    #[error(
        "feature {column} was complete at fit time but is missing at transform; set error_on_new(false) to tolerate new missingness"
    )]
    UnexpectedMissingness {
        /// The feature column that never carried a missing value at fit time.
        column: usize,
    },

    /// A conversion failure at the data boundary.
    #[error("conversion failure: {0}")]
    Conversion(String),

    /// An operation that requires at least one observation received none.
    #[error("operation `{operation}` requires at least one observation")]
    EmptyInput {
        /// The operation that required observations.
        operation: &'static str,
    },

    /// A non-finite numeric value received where finite values are required.
    #[error("non-finite value at position {position}")]
    NonFiniteValue {
        /// The zero-based position of the offending value.
        position: usize,
    },

    /// A probability outside the closed unit interval.
    #[error("probability at position {position} is outside [0, 1]")]
    ProbabilityOutOfRange {
        /// The zero-based position of the offending probability.
        position: usize,
    },

    /// A label observed outside an explicitly supplied label space.
    #[error("label {label} is outside the resolved label space")]
    UnknownLabel {
        /// The offending label.
        label: i64,
    },

    /// A tabular selection named a column the schema does not carry.
    #[error("unknown column `{column}`; {suggestion}")]
    UnknownColumn {
        /// The name (or position) that could not be resolved.
        column: String,
        /// A way to correct the selection.
        suggestion: String,
    },

    /// A tabular schema or selection carries the same column more than once.
    #[error("duplicate column `{column}`; {suggestion}")]
    DuplicateColumn {
        /// The duplicated column name.
        column: String,
        /// A way to resolve the duplication.
        suggestion: String,
    },

    /// A column's logical type cannot enter the requested conversion path.
    #[error(
        "unsupported column type `{logical_type}` for column `{column}`; {suggestion}"
    )]
    UnsupportedColumnType {
        /// The column whose type was rejected.
        column: String,
        /// The rejected logical type.
        logical_type: String,
        /// A conversion path the consumer can take.
        suggestion: String,
    },

    /// A nullable column carries a null the selected policy does not handle.
    #[error("column `{column}` contains a null value under the `{policy}` missing-value policy")]
    NullValueInColumn {
        /// The column carrying the null.
        column: String,
        /// The policy that rejected it.
        policy: &'static str,
    },

    /// A tabular batch's columns do not all have the same length.
    #[error(
        "column `{column}` has {found} rows but the batch has {expected} rows; all columns must be row-aligned"
    )]
    UnevenColumnLengths {
        /// The column whose length differs.
        column: String,
        /// The row count the batch expects.
        expected: usize,
        /// The row count the column carries.
        found: usize,
    },

    /// A schema and the data (or another schema) do not describe the same table.
    #[error("schema mismatch: expected {expected}, found {found}")]
    SchemaMismatch {
        /// What the receiving side expected.
        expected: String,
        /// What it actually received.
        found: String,
    },
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
