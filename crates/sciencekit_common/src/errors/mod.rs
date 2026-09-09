//! Central error taxonomy shared across the whole library (spec `error-model`).
//!
//! `SKError` names precisely the common failures every algorithm can produce.
//! Algorithm-specific error enums convert from it automatically via
//! [`From<SKError>`], keeping common errors identical across algorithms.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; the implementation lives in `error_kind.rs`.

mod error_kind;

pub use error_kind::SKError;

#[cfg(test)]
mod errors_tests;
