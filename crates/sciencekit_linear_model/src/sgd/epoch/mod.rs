//! The epoch drivers of the stochastic-gradient substrate
//! (spec `stochastic-gradient-solver`, decision 9).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod batch;
mod dense;
mod indexed;
mod sparse;

pub(crate) use batch::sk_run_batch_epoch;
pub(crate) use dense::sk_run_dense_epoch;
pub(crate) use sparse::sk_run_sparse_epoch;

#[cfg(test)]
mod epoch_tests;
