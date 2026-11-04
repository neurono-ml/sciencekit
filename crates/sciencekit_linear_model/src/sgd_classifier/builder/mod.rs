//! Pure dispatcher: the SGD classifier builder surface.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod construction;
mod setters;

pub use construction::SKSGDClassifierBuilder;
