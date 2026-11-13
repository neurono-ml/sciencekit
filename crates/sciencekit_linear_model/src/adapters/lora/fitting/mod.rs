//! The adapter optimization: dispatch, validation, gradients and the two
//! family-specific descents.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares submodules;
//! every implementation lives in its own file.

mod dispatch;
mod gradients;
mod linear;
mod logistic;
mod validation;
