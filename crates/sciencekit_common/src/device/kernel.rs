//! The closed kernel catalogue shared by device capabilities and dispatch
//! (spec `gpu-compute-contracts`).

use std::fmt;

/// The kernel families a device backend may implement.
///
/// The catalogue is closed: a backend either supports one of these kernels or
/// reports it as unsupported, and the planner filters inventory by the kernel
/// an operation requires.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SKGpuKernel {
    /// Pairwise distances between rows of a matrix.
    PairwiseDistance,
    /// Dense matrix multiplication (`C = α · A·B`).
    Gemm,
    /// Elementwise transformation of a matrix.
    Elementwise,
    /// Tree-ensemble prediction.
    TreePredict,
}

impl SKGpuKernel {
    /// Every kernel, in the documented order.
    pub const ALL: [SKGpuKernel; 4] = [
        SKGpuKernel::PairwiseDistance,
        SKGpuKernel::Gemm,
        SKGpuKernel::Elementwise,
        SKGpuKernel::TreePredict,
    ];

    /// The documented static name of this kernel.
    pub fn name(self) -> &'static str {
        match self {
            SKGpuKernel::PairwiseDistance => "pairwise-distance",
            SKGpuKernel::Gemm => "gemm",
            SKGpuKernel::Elementwise => "elementwise",
            SKGpuKernel::TreePredict => "tree-predict",
        }
    }
}

impl fmt::Display for SKGpuKernel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl std::str::FromStr for SKGpuKernel {
    type Err = crate::SKError;

    /// Parse the documented kernel name back into its kernel.
    fn from_str(name: &str) -> Result<Self, Self::Err> {
        SKGpuKernel::ALL
            .into_iter()
            .find(|kernel| kernel.name() == name)
            .ok_or_else(|| crate::SKError::Conversion(format!("unknown GPU kernel `{name}`")))
    }
}
