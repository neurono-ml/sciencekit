//! C-Support-Vector Classifier (`SKSVC`)
//! (spec `support-vector-machines`).
//!
//! Pure dispatcher module: implementations live in dedicated files.

mod builder;
mod core_implementation;
mod weights;

pub use builder::SKSVCBuilder;
pub use core_implementation::SKSVC;
pub use weights::{SKClassWeight, sk_resolve_class_weights, sk_validate_static_class_weight};

#[cfg(test)]
mod svc_builder_tests;
