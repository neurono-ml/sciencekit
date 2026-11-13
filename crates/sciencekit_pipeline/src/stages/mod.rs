//! The typed chain of configured stages.
//!
//! A **pure dispatcher** module: it declares the chain submodule and re-exports
//! its public trait.

mod chain;

pub use chain::SKConfiguredStages;

#[cfg(test)]
mod chain_tests;
