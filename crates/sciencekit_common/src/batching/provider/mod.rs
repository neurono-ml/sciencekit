//! Source providers (spec `streaming-source-provider`).
//!
//! A provider opens one streaming pass at a time. The feature-only and
//! supervised families mirror each other; a mutable reference to a concrete
//! source is itself a one-pass provider, while [`SKReopenableSource`] wraps a
//! fallible factory and opens a fresh source per pass.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod reopenable;
mod source_provider;
mod supervised_provider;

pub use reopenable::SKReopenableSource;
pub use source_provider::SKSourceProvider;
pub use supervised_provider::SKSupervisedSourceProvider;

#[cfg(test)]
mod provider_test_sources;
#[cfg(test)]
mod provider_tests;
#[cfg(test)]
mod supervised_provider_tests;
