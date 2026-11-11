//! The Arrow C Data Interface (spec `arrow-columnar-interop`).
//!
//! `unsafe` is confined to this module: exporting and importing the C
//! structures is the only place the adapter touches raw pointers, and the
//! release callbacks are owned by Arrow's own `FFI_ArrowArray`,
//! `FFI_ArrowSchema` and `FFI_ArrowArrayStream` drops. Safe entry points
//! transfer ownership of exporter-produced pairs and streams exactly once;
//! reassembling raw C structures via [`SKArrowFfiArray::from_parts`] or
//! [`SKArrowFfiStream::from_stream`] is `unsafe` because the caller asserts the
//! C ABI invariants.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

#![allow(unsafe_code)]

mod array_interface;
mod stream_interface;

pub use array_interface::{SKArrowFfiArray, sk_export_array_to_ffi, sk_import_array_from_ffi};
pub use stream_interface::{
    SKArrowFfiStream, sk_export_record_batch_reader_to_stream, sk_import_stream,
};

#[cfg(test)]
mod ffi_release_tests;
#[cfg(test)]
mod ffi_tests;
