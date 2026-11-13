//! `sciencekit_svm` — support-vector machines (`SKSVC`, `SKSVR` and
//! `SKLinearSVC`) (specs `support-vector-machines` and
//! `linear-support-vector-classifier`, PRD §12 Phase 3).
//!
//! Provides C-support-vector classification (`SKSVC`) with one-vs-one
//! multiclass, class weights and optional Platt + pairwise-coupling probability
//! calibration, alongside epsilon-insensitive support-vector regression
//! (`SKSVR`). Both share a deterministic pure-Rust second-order SMO solver, a
//! bounded LRU kernel-row cache and dense/sparse/mixed representation parity.
//! `SKLinearSVC` adds the LIBLINEAR-style linear counterpart: dual coordinate
//! descent for the hinge and squared-hinge losses over a dense or borrowed-CSR
//! design, one-vs-rest multiclass heads and an explicit convergence budget.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod kernel;
pub mod linear_svc;
pub mod probability;
pub mod smo;
pub mod svc;
pub mod svr;

pub use linear_svc::{
    SKLinearSVC, SKLinearSVCBuilder, SKLinearSvcLoss, sk_linear_svc_loss_from_code,
};
pub use sciencekit_math::{SKKernel, SKKernelGamma, SKKernelParameters};
pub use svc::{SKBinarySVCModel, SKClassWeight, SKSVC, SKSVCBuilder, SKSVCModel};
pub use svr::{SKSVR, SKSVRBuilder, SKSVRModel};

#[cfg(test)]
mod acceptance_tests;
#[cfg(test)]
mod concurrency_tests;
#[cfg(test)]
mod determinism_tests;
#[cfg(test)]
mod observability_and_execution_tests;
#[cfg(test)]
mod representation_fit_tests;
#[cfg(test)]
mod representation_predict_tests;
#[cfg(test)]
mod validation_tests;
