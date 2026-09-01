//! Mandatory builder foundation (spec `base-builders`, PRD §2.1, §5.3).
//!
//! Every public estimator, transformer and algorithm is constructed through a
//! typed builder ([`SKBuilder`]) whose direct constructor stays private. The
//! shared [`SKBuilderState`] carries the execution intent (defaulting to
//! `Automatic`) and composes the automatic execution decision, so algorithm
//! crates reuse the intent plumbing instead of re-implementing it.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod builder_state;
mod builder_trait;
mod reference_estimator;
mod validation;

#[cfg(test)]
mod builder_tests;

pub use builder_state::SKBuilderState;
pub use builder_trait::SKBuilder;
pub use reference_estimator::{SKReferenceEstimator, SKReferenceEstimatorBuilder};
pub use validation::sk_validate_hyperparameter;
