//! The device-buffer and transfer surface of the OpenCL backend (spec
//! `opencl-backend`, requirement *OpenCL buffers*).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; the allocation and the resident matrix live in their own files.

mod allocation;
mod resident;

pub use resident::SKOpenClBuffer;

#[cfg(test)]
pub(crate) use allocation::SKOpenClAllocation;
pub(crate) use resident::{allocate_resident, download_buffer, upload_owned, upload_view};

#[cfg(test)]
mod buffer_tests;
