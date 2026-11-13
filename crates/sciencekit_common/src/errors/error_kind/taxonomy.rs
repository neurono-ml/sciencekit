//! The central error enum.

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
    /// Input data uses a representation this algorithm does not support.
    #[error(
        "data representation `{representation}` is not supported by this algorithm; {suggestion}"
    )]
    UnsupportedRepresentation {
        /// The representation that was rejected.
        representation: &'static str,
        /// A conversion path the consumer can take.
        suggestion: &'static str,
    },
    /// An operation that an otherwise valid configuration cannot perform.
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
    /// An explicit execution mode incompatible with the declared access pattern.
    #[error("execution mode `{mode}` is incompatible with the declared access pattern `{pattern}`")]
    ExecutionModeIncompatible {
        /// The requested mode.
        mode: &'static str,
        /// The declared access pattern.
        pattern: &'static str,
    },
    /// A streaming batch hint whose double-buffered footprint exceeds available memory.
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
    /// Missingness met at transform in a feature complete at fit time under `error_on_new`.
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
    #[error("unsupported column type `{logical_type}` for column `{column}`; {suggestion}")]
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
    /// A requested device backend has no usable enumerated device or runtime.
    #[error("device backend `{backend}` is unavailable: {reason}")]
    DeviceUnavailable {
        /// The backend whose device could not be used.
        backend: &'static str,
        /// Why the device is unavailable.
        reason: String,
    },
    /// A device backend does not execute the scalar type an operation carries.
    #[error("device backend `{backend}` does not support the scalar type `{scalar_type}`")]
    UnsupportedScalarType {
        /// The backend that rejected the scalar type.
        backend: &'static str,
        /// The rejected scalar type, by its Rust name (`f32`, `f64`).
        scalar_type: &'static str,
    },
    /// A host↔device transfer failed after the device was selected.
    #[error("device transfer failed: {reason}")]
    DeviceTransferFailure {
        /// Why the transfer failed.
        reason: String,
    },
    /// A memory-mapped source names a file that does not exist.
    #[error("memory-mapped file `{path}` does not exist")]
    MissingMappedFile {
        /// The path that was not found.
        path: String,
    },
    /// A memory-mapped source could not read the file for lack of permission.
    #[error("permission denied reading memory-mapped file `{path}`: {reason}")]
    MappedFilePermissionDenied {
        /// The path whose access was denied.
        path: String,
        /// The operating-system explanation.
        reason: String,
    },
    /// A memory-mapped file carries a dtype the model scalar cannot view.
    #[error("memory-mapped file `{path}` carries dtype `{dtype}`; supported: {supported}")]
    UnsupportedMappedDtype {
        /// The file that declared the dtype.
        path: String,
        /// The dtype the file carries.
        dtype: String,
        /// The dtypes the opening path accepts.
        supported: String,
    },
    /// A raw or NumPy memory-mapped file declares a non-little-endian byte order.
    #[error(
        "memory-mapped file `{path}` is {endianness}-endian; only little-endian data is supported"
    )]
    NonLittleEndianMappedData {
        /// The file that declared the byte order.
        path: String,
        /// The declared byte order (`big`).
        endianness: String,
    },
    /// A memory-mapped file's byte length differs from its declared shape.
    #[error(
        "memory-mapped file `{path}` has {actual_bytes} bytes but the declared shape requires {expected_bytes} bytes"
    )]
    MappedLengthMismatch {
        /// The file whose length was checked.
        path: String,
        /// The byte count the declared shape requires.
        expected_bytes: u64,
        /// The byte count the file carries.
        actual_bytes: u64,
    },
    /// A memory-mapped file's header is malformed or truncated.
    #[error("malformed {format} header in memory-mapped file `{path}`: {reason}")]
    MalformedMappedHeader {
        /// The file whose header was parsed.
        path: String,
        /// The header format (`npy`, `arrow-ipc`, `raw-binary-descriptor`).
        format: &'static str,
        /// Why the header was rejected.
        reason: String,
    },
    /// A memory-mapped payload is not aligned for the target scalar type.
    #[error("memory-mapped file `{path}` payload is misaligned for {scalar_type}: {reason}")]
    MisalignedMappedData {
        /// The file whose payload is misaligned.
        path: String,
        /// The scalar type whose alignment is required.
        scalar_type: &'static str,
        /// The alignment diagnosis and the suggested remedy.
        reason: String,
    },
    /// A memory-mapped record batch cannot be read.
    #[error("memory-mapped file `{path}` cannot serve batch {batch}: {reason}")]
    UnreadableMappedBatch {
        /// The file that owns the batch.
        path: String,
        /// The zero-based batch index.
        batch: usize,
        /// Why the batch could not be read.
        reason: String,
    },
}
