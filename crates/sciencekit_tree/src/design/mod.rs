//! Representation-dispatched fit design (spec `data-view-boundary`).
//!
//! This is a **pure dispatcher** module: `mod.rs` declares the submodule and
//! re-exports its items; the implementation lives in `design_view`.

mod design_view;

pub(crate) use design_view::{SKDesign, sk_to_csc};

#[cfg(test)]
mod design_tests;
