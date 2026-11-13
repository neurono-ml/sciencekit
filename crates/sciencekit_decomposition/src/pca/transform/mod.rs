//! Projection (`transform`), combined `fit_transform` and reconstruction
//! (`inverse_transform`) for [`super::SKPCA`] and [`super::SKPCAModel`].
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares submodules;
//! every implementation lives in its own file.

mod projection;
mod reconstruction;
