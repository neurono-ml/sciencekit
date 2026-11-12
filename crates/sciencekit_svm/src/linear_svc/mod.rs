//! L2-regularized linear support-vector classifier (`SKLinearSVC`)
//! (spec `linear-support-vector-classifier`).
//!
//! Pure dispatcher module: implementations live in dedicated files.

mod builder;
mod config;
mod core_implementation;
mod design;

pub use builder::SKLinearSVCBuilder;
pub use config::{SKLinearSvcLoss, sk_linear_svc_loss_from_code};
pub use core_implementation::SKLinearSVC;

#[cfg(test)]
mod linear_svc_builder_tests;
#[cfg(test)]
mod linear_svc_config_tests;
#[cfg(test)]
mod linear_svc_design_tests;
