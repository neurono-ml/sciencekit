//! MiniBatchKMeans: per-center learning rates, deterministic reassignment,
//! `partial_fit` and multi-epoch `fit_streaming` (spec `mini-batch-k-means`).
//!
//! This is a **pure dispatcher** module: it only declares and re-exports its
//! submodules; every implementation lives in its own file.

mod assign;
mod builder;
mod config;
mod core_implementation;
mod fit;
mod init;
mod model;
mod partial_fit;
mod reassign;
mod state;
mod streaming;
mod update;
mod validation;

pub use crate::kmeans::SKKMeansInitialization;
pub use builder::SKMiniBatchKMeansBuilder;
pub use core_implementation::SKMiniBatchKMeans;
pub use model::SKMiniBatchKMeansModel;

#[cfg(test)]
mod minibatch_init_tests;
#[cfg(test)]
mod minibatch_partial_tests;
#[cfg(test)]
mod minibatch_reassign_tests;
#[cfg(test)]
mod minibatch_representation_tests;
#[cfg(test)]
mod minibatch_state_tests;
#[cfg(test)]
mod minibatch_streaming_tests;
#[cfg(test)]
mod minibatch_update_tests;
