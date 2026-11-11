//! C-Support-Vector Classifier (`SKSVC`)
//! (spec `support-vector-machines`).
//!
//! Pure dispatcher module: implementations live in dedicated files.

mod builder;
mod core_implementation;
mod fit;
mod model;
mod predict;
mod targets;
mod weights;

pub use builder::SKSVCBuilder;
pub use core_implementation::SKSVC;
pub use model::{SKBinarySVCModel, SKSVCModel};
pub use predict::sk_resolve_ovo_votes;
pub use weights::{SKClassWeight, sk_resolve_class_weights, sk_validate_static_class_weight};

#[cfg(test)]
mod svc_builder_tests;
#[cfg(test)]
mod svc_fit_tests;
#[cfg(test)]
mod svc_multiclass_tests;
#[cfg(test)]
mod svc_predict_tests;
