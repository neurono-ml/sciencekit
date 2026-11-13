//! Memory-mapped sources: raw binary, NumPy `.npy` and Arrow IPC (spec
//! `memory-mapped-sources`, feature `memmap`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares submodules and
//! re-exports their public items. Unsafe code is confined to the zero-copy
//! view helpers, each of which documents and validates its alignment and
//! lifetime preconditions.
//!
//! # File stability contract
//!
//! A mapped source reads directly from the operating system's mapping of the
//! file for as long as the source — and any row view borrowed from it — is
//! alive. The mapping is created read-only and sciencekit never writes to it,
//! but no portable mechanism can stop *another* process (or safe code holding
//! the same path) from truncating, rewriting or replacing the file meanwhile.
//!
//! The caller must therefore guarantee that the mapped file is not modified,
//! truncated or replaced while the source is alive; open a private snapshot or
//! copy when the file may change. Failure mode if the contract is broken:
//! on Unix, reading a page of a file that was truncated underneath the mapping
//! raises `SIGBUS`, and a concurrent rewrite yields torn or stale values. Both
//! are process-level faults, not catchable Rust errors; this is why the
//! contract is stated here rather than checked at runtime.

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

#[cfg(all(test, feature = "arrow"))]
mod arrow_ipc_layout_tests;
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
