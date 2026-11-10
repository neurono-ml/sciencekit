//! The central error enum and its convenience constructors.
//!
//! This is a **pure dispatcher** module for the central taxonomy: it declares
//! and re-exports the submodules; the enum lives in `taxonomy` and the
//! constructors in `constructors`.

mod constructors;
mod taxonomy;

pub use taxonomy::SKError;
