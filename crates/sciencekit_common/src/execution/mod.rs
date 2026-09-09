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

mod context;
mod modes;
mod plan;
mod resolve;

pub use context::SKExecutionContext;
pub use modes::{SKAccessPattern, SKExecutionMode};
pub use plan::SKExecutionPlan;
pub use resolve::sk_resolve_execution_plan;

#[cfg(test)]
mod execution_tests;
