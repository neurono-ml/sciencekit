//! Path-based, reopenable Arrow file sources (spec `arrow-columnar-interop`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares and re-exports
//! submodules; every implementation lives in its own file.

mod csv_reader;
mod format;
mod ipc_reader;
mod json_reader;
mod opener;
mod options;
mod parquet_reader;
mod projected_reader;
mod source;

pub use format::SKArrowFileFormat;
pub use source::SKArrowFileSource;

#[cfg(test)]
mod csv_tests;
#[cfg(test)]
mod file_tests;
#[cfg(test)]
mod ipc_tests;
#[cfg(test)]
mod json_tests;
#[cfg(test)]
mod large_file_tests;
#[cfg(test)]
mod parquet_tests;
#[cfg(test)]
mod projection_pushdown_tests;
#[cfg(test)]
mod projection_tests;
