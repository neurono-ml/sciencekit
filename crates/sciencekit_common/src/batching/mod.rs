//! Streaming batches (spec `streaming-batches`).
//!
//! An owned data block ([`SKDataBatch`]) that fully owns its data and can move
//! across threads; a sequential fallible source ([`SKLazySource`]) and an
//! abstract random-access source ([`SKMappableSource`]). The supervised
//! companions ([`SKSupervisedBatch`] / [`SKSupervisedLazySource`]) pair features
//! with targets for out-of-core supervised fitting. Memory-mapped
//! implementations belong to the interop layer, not here.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod data_batch;
mod lazy_source;
mod mappable_source;
mod supervised_batch;
mod supervised_source;

pub use data_batch::SKDataBatch;
pub use lazy_source::SKLazySource;
pub use mappable_source::SKMappableSource;
pub use supervised_batch::SKSupervisedBatch;
pub use supervised_source::SKSupervisedLazySource;

#[cfg(test)]
mod batching_tests;
