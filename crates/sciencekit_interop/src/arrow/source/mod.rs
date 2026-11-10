//! Streaming Arrow sources over record-batch readers (spec
//! `arrow-columnar-interop`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod batch_source;
mod cursor;
mod one_shot_source;
mod projection;

pub use batch_source::SKArrowRecordBatchSource;
pub use one_shot_source::SKArrowOneShotTabularSource;
pub(crate) use projection::sk_projection_indices;

#[cfg(test)]
mod source_tests;
