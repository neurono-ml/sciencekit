//! Class-weighting policy for the classifier (spec `decision-trees`).
//!
//! This is a **pure dispatcher** module: `mod.rs` declares the submodules and
//! re-exports their items; the implementation lives in `policy` and `expansion`.

mod expansion;
mod policy;

pub use expansion::sk_expand_class_weights;
pub use policy::SKClassWeight;

#[cfg(test)]
mod class_weight_tests;
