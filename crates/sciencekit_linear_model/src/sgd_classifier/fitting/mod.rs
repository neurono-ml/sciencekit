//! The classifier's fitting machinery: head creation, the dense/CSR epoch loop
//! and the single-pass stream.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod driver;
mod epochs;
mod state;
mod streaming;

pub(crate) use driver::{SKFeatureDesign, sk_run_head_epoch};
pub(crate) use epochs::sk_fit_epochs;
pub(crate) use state::{sk_create_states, sk_reported_parts};
pub(crate) use streaming::sk_fit_streaming;
