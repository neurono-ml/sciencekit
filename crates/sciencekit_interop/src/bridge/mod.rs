//! The async-to-sync bridge module (spec `dataset-interop-core`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod async_source;
mod async_stream;

pub use async_source::{SK_DEFAULT_ASYNC_BRIDGE_DEPTH, SKAsyncTabularSource};
pub use async_stream::SKAsyncTabularStream;

#[cfg(test)]
mod bridge_tests;
#[cfg(test)]
mod error_tests;
#[cfg(test)]
mod cancellation_tests;
#[cfg(test)]
mod test_streams;
