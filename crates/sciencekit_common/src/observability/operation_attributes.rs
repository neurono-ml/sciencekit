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

/// The bounded set of compute backends an operation may route to.
///
/// The first three kinds are host (CPU) backends; `OpenCl`, `Cuda` and `Rocm`
/// name device backends whose concrete implementations live in the future
/// `sciencekit_gpu_opencl`, `sciencekit_gpu_cuda` and `sciencekit_gpu_rocm`
/// crates. The enum is deliberately closed: the compiler forces every
/// exhaustive match in the workspace to name the device kinds too.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SKBackendKind {
    /// The pure-Rust default backend backed by `faer`.
    Faer,
    /// The GEMM-fallback backend backed by `matrixmultiply`.
    MatrixMultiply,
    /// The opt-in `blas-backend` backed by `ndarray-linalg`.
    NdArrayLinalg,
    /// A device backend driven through OpenCL (vendor-transparent).
    OpenCl,
    /// A device backend driven through CUDA.
    Cuda,
    /// A device backend driven through ROCm/HIP.
    Rocm,
}

impl SKBackendKind {
    /// Every backend kind, in the documented display order.
    pub const ALL: [SKBackendKind; 6] = [
        SKBackendKind::Faer,
        SKBackendKind::MatrixMultiply,
        SKBackendKind::NdArrayLinalg,
        SKBackendKind::OpenCl,
        SKBackendKind::Cuda,
        SKBackendKind::Rocm,
    ];

    /// Whether this kind is a host (CPU) backend.
    pub fn is_host(self) -> bool {
        !self.is_device()
    }

    /// Whether this kind is a device (GPU) backend.
    pub fn is_device(self) -> bool {
        matches!(
            self,
            SKBackendKind::OpenCl | SKBackendKind::Cuda | SKBackendKind::Rocm
        )
    }

    /// The documented static display name of this kind.
    ///
    /// Unlike [`Display`](std::fmt::Display), this returns a `&'static str`, so
    /// structured errors can carry the backend name without allocating.
    pub fn name(self) -> &'static str {
        match self {
            SKBackendKind::Faer => "faer",
            SKBackendKind::MatrixMultiply => "matrixmultiply",
            SKBackendKind::NdArrayLinalg => "ndarray-linalg",
            SKBackendKind::OpenCl => "opencl",
            SKBackendKind::Cuda => "cuda",
            SKBackendKind::Rocm => "rocm",
        }
    }
}

impl Default for SKBackendKind {
    /// The documented default backend ([`SKBackendKind::Faer`], spec
    /// `backend-selection`).
    fn default() -> Self {
        SKBackendKind::Faer
    }
}

impl fmt::Display for SKBackendKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for SKBackendKind {
    type Err = crate::SKError;

    /// Parse the documented display name back into its kind.
    fn from_str(name: &str) -> Result<Self, Self::Err> {
        SKBackendKind::ALL
            .into_iter()
            .find(|kind| kind.name() == name)
            .ok_or_else(|| crate::SKError::Conversion(format!("unknown backend kind `{name}`")))
    }
}

/// Structured attributes recorded on an operation span.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SKOperationAttributes {
    /// The operation kind (`fit`, `transform`, `predict`, ...).
    pub operation: SKOperationKind,
    /// The number of rows of the operation input.
    pub rows_count: usize,
    /// The number of columns of the operation input.
    pub columns_count: usize,
    /// The execution mode the operation resolved to.
    pub execution_mode: SKExecutionMode,
    /// The math backend the operation uses for heavy dense algebra.
    pub backend: SKBackendKind,
}
