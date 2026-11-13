//! Memory-mapped sources: raw binary, NumPy `.npy` and Arrow IPC (spec
//! `memory-mapped-sources`, feature `memmap`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares submodules and
//! re-exports their public items. Unsafe code is confined to the zero-copy
//! view helpers, each of which documents and validates its alignment and
//! lifetime preconditions.

mod alignment;
#[cfg(feature = "arrow")]
mod arrow_ipc_layout;
#[cfg(feature = "arrow")]
mod arrow_ipc_source;
mod descriptor;
mod mapping;
mod numpy_header;
mod numpy_source;
mod raw_binary_source;
mod sequential;

pub use descriptor::{SKRawBinaryDescriptor, SKRawBinaryDtype, SKRawBinaryEndianness};
pub use numpy_source::SKNumpyMappableSource;
pub use raw_binary_source::SKRawBinaryMappableSource;

#[cfg(feature = "arrow")]
pub use arrow_ipc_source::SKArrowIpcMappableSource;

#[cfg(test)]
mod arrow_ipc_tests;
#[cfg(test)]
mod concurrency_tests;
#[cfg(test)]
mod descriptor_tests;
#[cfg(test)]
mod hostile_input_tests;
#[cfg(test)]
mod numpy_tests;
#[cfg(test)]
mod raw_binary_tests;
#[cfg(test)]
pub(crate) mod test_fixtures;
