//! C-Support-Vector Classifier (`SKSVC`)
//! (spec `support-vector-machines`).
//!
//! Pure dispatcher module: implementations live in dedicated files.

mod binary_fit;
mod builder;
mod core_implementation;
mod decision;
mod export;
mod fit;
mod model;
mod predict;
mod targets;
mod weights;

pub(crate) use binary_fit::sk_extract_subset_features;
pub use builder::SKSVCBuilder;
pub use core_implementation::SKSVC;
pub use model::{SKBinarySVCModel, SKSVCModel};
pub use predict::sk_resolve_one_vs_one_votes;
pub use weights::{SKClassWeight, sk_resolve_class_weights, sk_validate_static_class_weight};

#[cfg(test)]
mod svc_builder_tests;
#[cfg(test)]
mod svc_decision_tests;
#[cfg(test)]
mod svc_fit_tests;
#[cfg(test)]
mod svc_multiclass_tests;
#[cfg(test)]
mod svc_predict_tests;
#[cfg(test)]
mod svc_probability_tests;
#[cfg(test)]
mod svc_weights_tests;

#[cfg(test)]
mod export_tests;
