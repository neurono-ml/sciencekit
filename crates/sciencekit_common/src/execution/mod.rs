//! Execution planning (spec `execution-planning`).
//!
//! Separates the consumer-declared **intent** ([`SKExecutionMode`]) from the
//! effective **plan** ([`SKExecutionPlan`]) resolved per operation. Resolution
//! is a pure function over an injectable [`SKExecutionContext`] — deterministic
//! and testable without a real machine. An explicitly requested mode
//! incompatible with the algorithm's declared access pattern fails with a
//! specific error; automatic intent never does.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod batch_guard;
mod context;
mod modes;
mod plan;
mod resolve;
mod streaming_driver;

pub use batch_guard::sk_check_batch_buffer_guard;
pub use context::SKExecutionContext;
pub use modes::{SKAccessPattern, SKExecutionMode};
pub use plan::SKExecutionPlan;
pub use resolve::sk_resolve_execution_plan;
pub use streaming_driver::{
    SKStreamDecision, SKStreamingPassPolicy, sk_run_streaming_driver, sk_run_streaming_provider,
    sk_run_supervised_streaming_driver, sk_run_supervised_streaming_provider,
};

#[cfg(test)]
mod execution_tests;
#[cfg(test)]
mod sparse_guard_tests;
#[cfg(test)]
mod streaming_driver_batch_guard_tests;
#[cfg(test)]
mod streaming_driver_sparse_tests;
#[cfg(test)]
mod streaming_driver_tests;
