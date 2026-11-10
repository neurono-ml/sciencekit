//! The capabilities a device reports to the execution planner.
//!
//! A capability summary is a bounded, comparable value: the planner filters
//! devices by the scalar types and kernels an operation requires without ever
//! touching a GPU runtime, so planning stays pure and testable.

use super::kernel::SKGpuKernel;

/// The capabilities one device advertises.
///
/// Every capability defaults to `false` — a device must opt into what it
/// supports, so an unknown backend can never be mistaken for a capable one.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct SKDeviceCapabilities {
    /// Whether the device executes 32-bit floating-point kernels.
    pub supports_f32: bool,
    /// Whether the device executes 64-bit floating-point kernels.
    pub supports_f64: bool,
    /// Whether the backend exposes a mapped (zero-copy) host↔device view.
    pub supports_mapped_memory: bool,
    /// The kernels the backend implements on this device.
    pub supported_kernels: Vec<SKGpuKernel>,
}

impl SKDeviceCapabilities {
    /// Build a capability summary from its scalar and memory flags; the kernel
    /// list starts empty and is filled by [`SKDeviceCapabilities::with_kernels`].
    pub fn new(supports_f32: bool, supports_f64: bool, supports_mapped_memory: bool) -> Self {
        SKDeviceCapabilities {
            supports_f32,
            supports_f64,
            supports_mapped_memory,
            supported_kernels: Vec::new(),
        }
    }

    /// Declare the kernels the device implements.
    pub fn with_kernels(mut self, kernels: &[SKGpuKernel]) -> Self {
        self.supported_kernels = kernels.to_vec();
        self
    }

    /// Whether the device implements `kernel`.
    pub fn supports_kernel(&self, kernel: SKGpuKernel) -> bool {
        self.supported_kernels.contains(&kernel)
    }

    /// Whether the device supports the scalar type named by its Rust name
    /// (`"f32"` or `"f64"`).
    ///
    /// An unknown scalar name is reported as unsupported rather than as an
    /// error: the question is a capability query, and the caller names the
    /// scalar type it carries.
    pub fn supports_scalar_type(&self, scalar_type: &str) -> bool {
        match scalar_type {
            "f32" => self.supports_f32,
            "f64" => self.supports_f64,
            _ => false,
        }
    }
}
