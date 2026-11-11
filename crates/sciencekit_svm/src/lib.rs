//! `sciencekit_svm` — kernel support-vector machines (`SKSVC` and `SKSVR`)
//! (spec `support-vector-machines`, PRD §12 Phase 3).
//!
//! Provides C-support-vector classification (`SKSVC`) with one-vs-one
//! multiclass, class weights and optional Platt + pairwise-coupling probability
//! calibration, alongside epsilon-insensitive support-vector regression
//! (`SKSVR`). Both share a deterministic pure-Rust second-order SMO solver, a
//! bounded LRU kernel-row cache and dense/sparse/mixed representation parity.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod kernel;
pub mod probability;
pub mod smo;
pub mod svc;
pub mod svr;

pub use sciencekit_math::{SKKernel, SKKernelGamma, SKKernelParameters};
pub use svc::{SKBinarySVCModel, SKClassWeight, SKSVC, SKSVCBuilder, SKSVCModel};
pub use svr::{SKSVR, SKSVRBuilder, SKSVRModel};

#[cfg(test)]
mod representation_fit_tests;
#[cfg(test)]
mod representation_predict_tests;
