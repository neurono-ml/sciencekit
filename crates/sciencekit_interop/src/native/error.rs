//! Typed failures produced by the native import trust boundary
//! (spec `native-model-import`, requirement "Validation and concurrency").

use sciencekit_common::{SKError, SKTensorDataType};

/// Why a model file or state could not become a native model.
///
/// The variants are deliberately finer-grained than [`SKError`]: an untrusted
/// byte slice can lie about its container (magic, version, length) before it can
/// lie about tensors, and the caller must be able to tell those stages apart.
/// Every variant is produced before a model exists, so a failed conversion never
/// leaves a partially initialized value behind.
#[derive(Debug)]
pub enum SKNativeImportError {
    /// The bytes do not start with a recognized container header.
    WrongMagic {
        /// The header this reader expected.
        expected: &'static str,
        /// The first bytes actually seen, at most eight.
        found: Vec<u8>,
    },
    /// The container declares a format major version this build cannot read.
    UnknownFormatVersion {
        /// The container family (`safetensors` or `json`).
        format: &'static str,
        /// The version found in the file.
        found: u64,
        /// The version this build reads.
        supported: u64,
    },
    /// The container ends before the bytes it declares.
    TruncatedFile {
        /// The container family (`safetensors` or `json`).
        format: &'static str,
        /// The number of bytes the header declares.
        needed: u64,
        /// The number of bytes actually available.
        available: u64,
    },
    /// A tensor the estimator requires is absent from the state.
    MissingTensor {
        /// The stable tensor name.
        name: String,
    },
    /// A tensor the estimator does not know is present in the state.
    ///
    /// Extra tensors are rejected rather than ignored: an unknown tensor means
    /// the file was produced by a different estimator or a newer format, and
    /// silently dropping it would hide that fact.
    UnexpectedTensor {
        /// The unexpected tensor name.
        name: String,
    },
    /// A tensor carries the wrong number of values for its declared shape.
    ShapeMismatch {
        /// The tensor name.
        tensor: String,
        /// The shape the importer needs.
        expected: Vec<usize>,
        /// The shape the state carries.
        found: Vec<usize>,
    },
    /// A tensor carries a different element type than the estimator expects.
    DtypeMismatch {
        /// The tensor name.
        tensor: String,
        /// The element type the importer needs.
        expected: SKTensorDataType,
        /// The element type the state carries.
        found: SKTensorDataType,
    },
    /// The state names an estimator no native importer covers.
    UnsupportedFamily {
        /// The estimator name exactly as the state spells it.
        estimator_name: String,
    },
    /// The graph uses a construct outside the documented subset.
    UnsupportedConstruct {
        /// The construct name (an operator, a node shape or a graph feature).
        construct: String,
        /// Why it is outside the subset.
        reason: String,
    },
    /// The central library error, propagated unchanged.
    Central(SKError),
}

impl core::fmt::Display for SKNativeImportError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::WrongMagic { expected, found } => write!(
                formatter,
                "wrong container header: expected {expected}, found the bytes {found:02x?}"
            ),
            Self::UnknownFormatVersion {
                format,
                found,
                supported,
            } => write!(
                formatter,
                "the {format} container declares format version {found}, but this build reads version {supported}"
            ),
            Self::TruncatedFile {
                format,
                needed,
                available,
            } => write!(
                formatter,
                "the {format} container declares {needed} bytes but only {available} are available"
            ),
            Self::MissingTensor { name } => {
                write!(
                    formatter,
                    "the state does not carry the required tensor `{name}`"
                )
            }
            Self::UnexpectedTensor { name } => write!(
                formatter,
                "the state carries the unexpected tensor `{name}`"
            ),
            Self::ShapeMismatch {
                tensor,
                expected,
                found,
            } => write!(
                formatter,
                "tensor `{tensor}` has shape {found:?}, the importer needs {expected:?}"
            ),
            Self::DtypeMismatch {
                tensor,
                expected,
                found,
            } => write!(
                formatter,
                "tensor `{tensor}` uses {}, the importer needs {}",
                found.as_str(),
                expected.as_str()
            ),
            Self::UnsupportedFamily { estimator_name } => write!(
                formatter,
                "no native importer covers the estimator `{estimator_name}`"
            ),
            Self::UnsupportedConstruct { construct, reason } => {
                write!(
                    formatter,
                    "`{construct}` is outside the native subset: {reason}"
                )
            }
            Self::Central(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for SKNativeImportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Central(error) => Some(error),
            _ => None,
        }
    }
}

impl From<SKError> for SKNativeImportError {
    fn from(error: SKError) -> Self {
        match error {
            SKError::ShapeMismatch { expected, found } => SKNativeImportError::ShapeMismatch {
                tensor: "state".to_string(),
                expected,
                found,
            },
            SKError::Io(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => {
                SKNativeImportError::TruncatedFile {
                    format: "model",
                    needed: 0,
                    available: 0,
                }
            }
            other => SKNativeImportError::Central(other),
        }
    }
}
