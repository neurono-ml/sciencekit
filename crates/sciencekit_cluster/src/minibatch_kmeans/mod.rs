//! MiniBatchKMeans: per-center learning rates, deterministic reassignment,
//! `partial_fit` and multi-epoch `fit_streaming` (spec `mini-batch-k-means`).
//!
//! This is a **pure dispatcher** module: it only declares and re-exports its
//! submodules; every implementation lives in its own file.

mod builder;
mod config;
mod core_implementation;
mod model;
mod state;

pub use crate::kmeans::SKKMeansInitialization;
pub use builder::SKMiniBatchKMeansBuilder;
pub use core_implementation::SKMiniBatchKMeans;
pub use model::SKMiniBatchKMeansModel;

#[cfg(test)]
mod minibatch_state_tests;
