//! L2-regularized linear support-vector classifier (`SKLinearSVC`)
//! (spec `linear-support-vector-classifier`).
//!
//! Pure dispatcher module: implementations live in dedicated files.

mod builder;
mod config;
mod core_implementation;
mod decision;
mod design;
mod fit;
mod model;
mod predict;
mod solver;
mod validation;

pub use builder::SKLinearSVCBuilder;
pub use config::{SKLinearSvcLoss, sk_linear_svc_loss_from_code};
pub use core_implementation::SKLinearSVC;
pub use model::{SKLinearSVCModel, SKLinearSvcHead};

#[cfg(test)]
mod acceptance_tests;
#[cfg(test)]
mod linear_svc_builder_tests;
#[cfg(test)]
mod linear_svc_concurrency_tests;
#[cfg(test)]
mod linear_svc_config_tests;
#[cfg(test)]
mod linear_svc_design_tests;
#[cfg(test)]
mod linear_svc_fit_tests;
#[cfg(test)]
mod linear_svc_multiclass_tests;
#[cfg(test)]
mod linear_svc_predict_tests;
#[cfg(test)]
mod linear_svc_representation_tests;
#[cfg(test)]
mod linear_svc_solver_tests;
#[cfg(test)]
mod linear_svc_validation_tests;
#[cfg(test)]
mod linear_svc_weights_tests;
