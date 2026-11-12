//! Interop conversion observability (spec `dataset-interop-core`).
//!
//! A conversion span records the engine source kind, the batch shape and the
//! missing-value policy, so traces remain useful across adapters.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod conversion_span;
mod source_kind;

pub use conversion_span::{SKInteropConversionAttributes, sk_run_interop_conversion};
pub use source_kind::SKInteropSourceKind;

#[cfg(test)]
mod observability_tests;
