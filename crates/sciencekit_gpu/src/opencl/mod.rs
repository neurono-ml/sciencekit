//! The OpenCL implementation of the device contracts (spec `opencl-backend`).
//!
//! The module is always compiled so planning can construct a fallback backend
//! and report an honest reason; the device machinery (session, buffers, kernels,
//! programs) is compiled only behind the `opencl` feature. The feature is off by
//! default, so the default build pulls in no OpenCL runtime and calls no OpenCL
//! symbol.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.
#[cfg(feature = "opencl")]
mod buffer;
#[cfg(feature = "opencl")]
mod kernels;
#[cfg(feature = "opencl")]
mod session;

mod availability;
mod backend;
mod device;
mod observability;
pub use backend::{SKOpenClBackend, SKOpenClBackendBuilder, SKOpenClTransferReport};
pub use device::{sk_opencl_devices, sk_register_opencl_device_inventory};

#[cfg(feature = "opencl")]
pub use buffer::SKOpenClBuffer;
#[cfg(feature = "opencl")]
pub use kernels::{SK_OPENCL_TREE_NODE_COLUMNS, SKOpenClDistanceMetric};
