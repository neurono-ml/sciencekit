//! Scalar typing: the single sealed floating-point bound used by every
//! continuous API across the library (spec `scalar-typing`).
//!
//! `SKFloat` aggregates the bounds required for numeric computation
//! (`num_traits::Float` for arithmetic), thread transfer (`Send + Sync`) and
//! static dispatch (`'static`). It is **sealed**: only the standard
//! floating-point types may implement it, and integers do not satisfy it.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; the implementation lives in `definition.rs`.

mod definition;

pub use definition::SKFloat;

#[cfg(test)]
mod sk_float_tests;
