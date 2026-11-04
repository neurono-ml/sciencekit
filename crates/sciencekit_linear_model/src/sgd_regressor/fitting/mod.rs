//! The regressor's fitting machinery: state creation, the dense/CSR epoch loop
//! and the single-pass stream.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod design;
mod epochs;
mod state;
mod streaming;

pub(crate) use design::{SKRegressorDesign, sk_run_design_epoch};
pub(crate) use epochs::sk_fit_epochs;
pub(crate) use state::{sk_create_state, sk_reported_parts};
pub(crate) use streaming::sk_fit_streaming;
