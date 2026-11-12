//! Representation-dispatched fit design (spec `data-view-boundary`).
//!
//! This is a **pure dispatcher** module: `mod.rs` declares the submodule and
//! re-exports its items; the implementation lives in `design_view` and
//! `conversion`.

mod conversion;
mod design_view;

pub(crate) use conversion::{sk_sparse_row_value, sk_to_csc};
pub(crate) use design_view::SKDesign;

#[cfg(test)]
mod design_tests;
