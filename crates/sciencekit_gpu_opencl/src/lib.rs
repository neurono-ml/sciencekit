//! `sciencekit_gpu_opencl` — the OpenCL 3.0 device backend of sciencekit
//! (spec `opencl-backend`).
//!
//! The backend implements the contracts defined in `sciencekit_gpu`, so it
//! cannot introduce a second device vocabulary: this crate is the vendor-facing
//! package and feature name, and every public type it exposes *is* the
//! `sciencekit_gpu` type. The implementation lives in `sciencekit_gpu` behind
//! its `opencl` feature because the contract traits are defined there; a
//! separate implementation crate depending on those traits would create a
//! package cycle.
//!
//! With the `gpu-opencl` feature off this crate exposes nothing and depends on
//! nothing; with it on it re-exports the OpenCL backend and its kernel
//! catalogue.

#![warn(missing_docs)]

#[cfg(feature = "gpu-opencl")]
pub use sciencekit_gpu::opencl::{
    SK_OPENCL_TREE_NODE_COLUMNS, SKOpenClBackend, SKOpenClBackendBuilder, SKOpenClBuffer,
    SKOpenClDistanceMetric, SKOpenClTransferReport, sk_opencl_devices,
    sk_register_opencl_device_inventory,
};

#[cfg(all(test, feature = "gpu-opencl"))]
mod lib_tests;
