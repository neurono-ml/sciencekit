//! Native ONNX graph import.
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares submodules and
//! re-exports their public items; every implementation lives in its own file.

mod error;
mod importer;
mod state;
pub use error::SKOnnxNativeError;
pub use importer::SKOnnxNativeImporter;

#[cfg(test)]
mod native_coverage_tests;
#[cfg(test)]
mod native_external_fixture_tests;
#[cfg(test)]
mod native_forest_tests;
#[cfg(test)]
mod native_import_tests;
#[cfg(test)]
mod native_validation_tests;
