//! Generic prefetch streaming drivers (spec `streaming-executor`).
//!
//! Consumes a sequential source through a dedicated I/O thread into a bounded
//! double buffer, handing each owned batch to an algorithm-supplied
//! `update(batch, parallelism, state)` step that runs on the compute side.
//! Input and computation overlap (the reader fetches batch k+1 while batch k
//! computes), bounded by the plan's buffer depth; batches are delivered in
//! order, fully owned.
//!
//! Two families of entry points share one prefetch engine: the one-pass
//! [`sk_run_streaming_driver`] / [`sk_run_supervised_streaming_driver`]
//! adapters, and the provider-based [`sk_run_streaming_provider`] /
//! [`sk_run_supervised_streaming_provider`] which open one source per pass.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod channel;
mod entry_points;
mod pass_policy;
mod prefetch;
mod provider_entry_points;

pub use entry_points::{sk_run_streaming_driver, sk_run_supervised_streaming_driver};
pub use pass_policy::SKStreamingPassPolicy;
pub use prefetch::SKStreamDecision;
pub use provider_entry_points::{sk_run_streaming_provider, sk_run_supervised_streaming_provider};

#[cfg(test)]
mod provider_entry_point_test_sources;
#[cfg(test)]
mod provider_entry_points_tests;
#[cfg(test)]
mod supervised_provider_entry_points_tests;
