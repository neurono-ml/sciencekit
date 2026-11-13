//! Typed failures produced while converting an ONNX graph into a native model
//! (spec `native-model-import`, requirement "ONNX native import").

use sciencekit_interop::native::SKNativeImportError;

use crate::import::SKOnnxImportError;

/// Why an ONNX graph could not become a native model.
#[derive(Debug)]
pub enum SKOnnxNativeError {
    /// A node uses a construct outside the documented native subset.
    UnsupportedOperator {
        /// The operator name exactly as the model spells it, or the graph
        /// construct that could not be mapped.
        operator: String,
        /// Why the construct is outside the subset.
        reason: String,
    },
    /// The graph names an estimator no native ONNX importer covers.
    UnsupportedEstimator {
        /// The estimator name carried in the ONNX metadata.
        estimator_name: String,
    },
    /// The graph is structurally valid but cannot be mapped unambiguously.
    InvalidGraph {
        /// What the mapper needed and what it found.
        detail: String,
    },
    /// The reconstructed state did not survive the native importer's validation.
    Native(SKNativeImportError),
    /// The ONNX loader rejected the bytes.
    Onnx(SKOnnxImportError),
}

impl core::fmt::Display for SKOnnxNativeError {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnsupportedOperator { operator, reason } => write!(
                formatter,
                "`{operator}` is outside the native ONNX subset: {reason}"
            ),
            Self::UnsupportedEstimator { estimator_name } => write!(
                formatter,
                "no native ONNX importer covers the estimator `{estimator_name}`"
            ),
            Self::InvalidGraph { detail } => {
                write!(formatter, "the ONNX graph cannot be mapped: {detail}")
            }
            Self::Native(error) => write!(formatter, "{error}"),
            Self::Onnx(error) => write!(formatter, "{error}"),
        }
    }
}

impl std::error::Error for SKOnnxNativeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Native(error) => Some(error),
            Self::Onnx(error) => Some(error),
            _ => None,
        }
    }
}

impl From<SKOnnxImportError> for SKOnnxNativeError {
    fn from(error: SKOnnxImportError) -> Self {
        SKOnnxNativeError::Onnx(error)
    }
}

impl From<SKNativeImportError> for SKOnnxNativeError {
    fn from(error: SKNativeImportError) -> Self {
        SKOnnxNativeError::Native(error)
    }
}
