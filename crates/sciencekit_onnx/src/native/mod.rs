//! ONNX to native model conversion (spec `native-model-import`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares submodules and
//! re-exports their public items; every implementation lives in its own file.

#[cfg(test)]
mod native_coverage_tests;
#[cfg(test)]
mod native_external_fixture_tests;
#[cfg(test)]
mod native_import_tests;

mod error;
mod importer;
mod state;
pub use error::SKOnnxNativeError;
pub use importer::SKOnnxNativeImporter;
