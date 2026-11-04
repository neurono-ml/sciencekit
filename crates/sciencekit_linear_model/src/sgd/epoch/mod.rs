//! The epoch drivers of the stochastic-gradient substrate
//! (spec `stochastic-gradient-solver`, decision 9).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod batch;
mod block;
mod dense;
mod indexed;
mod sparse;

// The drivers are consumed by the estimators in this change.
pub(crate) use batch::sk_run_batch_epoch;
pub(crate) use block::sk_run_block_epoch;
pub(crate) use dense::sk_run_dense_epoch;
pub(crate) use sparse::sk_run_sparse_epoch;

#[cfg(test)]
mod batch_tests;
#[cfg(test)]
mod dense_tests;
#[cfg(test)]
mod shape_tests;
#[cfg(test)]
mod sparse_tests;
#[cfg(test)]
mod test_helpers;
