//! Typed failures produced while loading and running an ONNX model.

use sciencekit_common::SKError;

/// Why an ONNX model could not be loaded or executed.
///
/// The variants are deliberately finer-grained than [`SKError`]: callers need
/// to distinguish a wrong opset from an unsupported operator or a missing
/// initializer, and the message must name the offending construct.
#[derive(Debug)]
pub enum SKOnnxImportError {
    /// The bytes are not a decodable `ModelProto`.
    InvalidModel {
        /// The decoder's diagnosis.
        detail: String,
    },
    /// The graph's data dependencies form a cycle.
    GraphCycle {
        /// The node labels along the detected cycle.
        path: Vec<String>,
    },
    /// The model targets an operator set the importer does not implement.
    UnsupportedOpset {
        /// The operator-set domain (`""` is the default ONNX domain).
        domain: String,
        /// The version the model declares.
        declared: i64,
        /// The newest version this importer understands.
        supported: i64,
    },
    /// A node uses an operator outside the documented subset.
    UnsupportedOperator {
        /// The operator-set domain.
        domain: String,
        /// The operator name exactly as the model spells it.
        operator: String,
        /// The node's zero-based position in the graph.
        node_index: usize,
    },
    /// A node reads a value no initializer or earlier node defines.
    MissingInitializer {
        /// The missing tensor name.
        name: String,
        /// The node that reads it.
        node_index: usize,
    },
    /// An initializer stores its payload outside the model file.
    ExternalData {
        /// The tensor whose data lives in an external file.
        tensor: String,
    },
    /// The model interface differs from what a dense predictor can consume.
    ShapeMismatch {
        /// The dimensions the predictor needs.
        expected: Vec<usize>,
        /// The dimensions the model received or declared.
        found: Vec<usize>,
    },
    /// The model or the input uses a dtype outside the dense numeric subset.
    UnsupportedDtype {
        /// The element type name.
        element_type: String,
    },
    /// The model does not describe a usable regression or classification head.
    UnsupportedOutput {
        /// Why the declared outputs are ambiguous.
        detail: String,
    },
    /// An operator received operands or attributes it cannot execute.
    InvalidOperator {
        /// The operator name.
        operator: String,
        /// What went wrong.
        detail: String,
    },
    /// The central library error, propagated unchanged.
    Central(SKError),
}

impl core::fmt::Display for SKOnnxImportError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidModel { detail } => {
                write!(f, "the bytes are not a valid ONNX model: {detail}")
            }
            Self::GraphCycle { path } => {
                write!(f, "the graph data dependencies form a cycle: {}", path.join(" -> "))
            }
            Self::UnsupportedOpset {
                domain,
                declared,
                supported,
            } => write!(
                f,
                "operator set {domain:?} version {declared} is not supported; this importer understands versions up to {supported}"
            ),
            Self::UnsupportedOperator {
                domain,
                operator,
                node_index,
            } => write!(
                f,
                "node {node_index} uses the unsupported operator {domain:?}::{operator:?}"
            ),
            Self::MissingInitializer { name, node_index } => write!(
                f,
                "node {node_index} reads `{name}`, which no initializer or earlier node defines"
            ),
            Self::ExternalData { tensor } => write!(
                f,
                "initializer `{tensor}` stores its payload in an external file, which this importer does not read"
            ),
            Self::ShapeMismatch { expected, found } => {
                write!(f, "shape mismatch: expected {expected:?}, found {found:?}")
            }
            Self::UnsupportedDtype { element_type } => {
                write!(
                    f,
                    "the element type `{element_type}` is outside the dense numeric subset"
                )
            }
            Self::UnsupportedOutput { detail } => {
                write!(f, "the model declares no usable prediction head: {detail}")
            }
            Self::InvalidOperator { operator, detail } => {
                write!(f, "operator {operator} could not execute: {detail}")
            }
            Self::Central(error) => write!(f, "{error}"),
        }
    }
}

impl std::error::Error for SKOnnxImportError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Central(error) => Some(error),
            _ => None,
        }
    }
}

impl From<SKError> for SKOnnxImportError {
    fn from(error: SKError) -> Self {
        SKOnnxImportError::Central(error)
    }
}

impl From<serde_onnx::proto::ProtoCodecError> for SKOnnxImportError {
    fn from(error: serde_onnx::proto::ProtoCodecError) -> Self {
        use serde_onnx::proto::ProtoCodecError;
        match error {
            ProtoCodecError::UnsupportedIrVersion {
                declared,
                supported,
            } => SKOnnxImportError::UnsupportedOpset {
                domain: "ir_version".to_string(),
                declared,
                supported,
            },
            ProtoCodecError::UnsupportedOpsetVersion {
                domain,
                declared,
                supported,
            } => SKOnnxImportError::UnsupportedOpset {
                domain,
                declared,
                supported,
            },
            other => SKOnnxImportError::InvalidModel {
                detail: other.to_string(),
            },
        }
    }
}
