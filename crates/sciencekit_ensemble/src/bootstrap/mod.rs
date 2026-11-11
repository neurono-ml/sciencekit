//! Bootstrap sampling and out-of-bag bookkeeping for ensemble estimators
//! (spec `random-forests`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod maximum_samples;
mod oob;
mod sampler;

pub use maximum_samples::SKMaximumSamples;
#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use oob::sk_out_of_bag_rows;
#[cfg_attr(not(test), allow(unused_imports))]
pub(crate) use sampler::{
    sk_bootstrap_seed, sk_gather_dense_rows, sk_gather_sparse_rows, sk_sample_indices, sk_tree_seed,
};

#[cfg(test)]
mod oob_tests;
#[cfg(test)]
mod sampler_tests;
