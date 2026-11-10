//! The capabilities a device reports to the execution planner.
//!
//! A capability summary is a bounded, comparable value: the planner filters
//! devices by the scalar types and kernels an operation requires without ever
//! touching a GPU runtime, so planning stays pure and testable.

/// The capabilities one device advertises.
///
/// Every capability defaults to `false` — a device must opt into what it
/// supports, so an unknown backend can never be mistaken for a capable one.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SKDeviceCapabilities {
    /// Whether the device executes 32-bit floating-point kernels.
    pub supports_f32: bool,
    /// Whether the device executes 64-bit floating-point kernels.
    pub supports_f64: bool,
    /// Whether the backend exposes a mapped (zero-copy) host↔device view.
    pub supports_mapped_memory: bool,
}

impl SKDeviceCapabilities {
    /// Build a capability summary from its three flags.
    pub fn new(supports_f32: bool, supports_f64: bool, supports_mapped_memory: bool) -> Self {
        SKDeviceCapabilities {
            supports_f32,
            supports_f64,
            supports_mapped_memory,
        }
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
