//! Generic prefetch streaming drivers (spec `streaming-executor`).
//!
//! Consumes a sequential source through a dedicated I/O thread into a bounded
//! double buffer, handing each owned batch to an algorithm-supplied
//! `update(batch, parallelism, state)` step that runs on the compute side.
//! Input and computation overlap (the reader fetches batch k+1 while batch k
//! computes), bounded by the plan's buffer depth; batches are delivered in
//! order, fully owned.
//!
//! Two public entry points share one prefetch engine: [`sk_run_streaming_driver`]
//! for feature-only [`SKLazySource`](crate::SKLazySource) blocks and
//! [`sk_run_supervised_streaming_driver`] for the supervised
//! [`SKSupervisedLazySource`](crate::SKSupervisedLazySource) blocks (features plus
//! targets).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod channel;
mod entry_points;
mod prefetch;

pub use entry_points::{sk_run_streaming_driver, sk_run_supervised_streaming_driver};
pub use prefetch::SKStreamDecision;
