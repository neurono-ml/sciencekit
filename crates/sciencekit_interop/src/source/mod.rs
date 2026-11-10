//! Tabular sources and their streaming-provider bridge (spec
//! `dataset-interop-core`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod in_memory_source;
mod lazy_source;
mod provider;
mod tabular_source;

pub use in_memory_source::SKInMemoryTabularSource;
pub use lazy_source::{SKSupervisedTabularLazySource, SKTabularLazySource};
pub use provider::SKTabularSourceProvider;
pub use tabular_source::SKTabularSource;

#[cfg(test)]
pub(crate) mod test_fixtures;
#[cfg(test)]
mod provider_tests;
#[cfg(test)]
mod source_tests;
