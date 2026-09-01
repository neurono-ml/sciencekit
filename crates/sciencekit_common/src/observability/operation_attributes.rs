//! Structured operation attributes for observability spans (spec `observability`).
//!
//! Bounded concepts are modelled as enums, never as free-form strings, so the
//! compiler rejects invalid values (e.g. a misspelled operation or backend)
//! instead of letting them surface at runtime.

use std::fmt;

use crate::execution::SKExecutionMode;

/// The bounded set of library operations that emit observability spans.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKOperationKind {
    /// Fit a model on data.
    Fit,
    /// Incrementally fit a model on a batch (streaming).
    PartialFit,
    /// Fit a transformer and transform the same data.
    FitTransform,
    /// Transform data with a fitted transformer.
    Transform,
    /// Predict targets for new data.
    Predict,
    /// Fit a model and predict on the same data.
    FitPredict,
    /// Score a fitted model against data.
    Score,
}

impl fmt::Display for SKOperationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            SKOperationKind::Fit => "fit",
            SKOperationKind::PartialFit => "partial_fit",
            SKOperationKind::FitTransform => "fit_transform",
            SKOperationKind::Transform => "transform",
            SKOperationKind::Predict => "predict",
            SKOperationKind::FitPredict => "fit_predict",
            SKOperationKind::Score => "score",
        };
        f.write_str(name)
    }
}

/// The bounded set of math backends an operation may route to.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKBackendKind {
    /// The pure-Rust default backend backed by `faer`.
    Faer,
    /// The GEMM-fallback backend backed by `matrixmultiply`.
    MatrixMultiply,
    /// The opt-in `blas-backend` backed by `ndarray-linalg`.
    NdArrayLinalg,
}

impl fmt::Display for SKBackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            SKBackendKind::Faer => "faer",
            SKBackendKind::MatrixMultiply => "matrixmultiply",
            SKBackendKind::NdArrayLinalg => "ndarray-linalg",
        };
        f.write_str(name)
    }
}

/// Structured attributes recorded on an operation span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SKOperationAttributes {
    /// The operation kind (`fit`, `transform`, `predict`, ...).
    pub operation: SKOperationKind,
    /// The number of rows of the operation input.
    pub rows: usize,
    /// The number of columns of the operation input.
    pub columns: usize,
    /// The execution mode the operation resolved to.
    pub execution_mode: SKExecutionMode,
    /// The math backend the operation uses for heavy dense algebra.
    pub backend: SKBackendKind,
}
