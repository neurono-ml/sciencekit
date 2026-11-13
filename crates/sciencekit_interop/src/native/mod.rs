//! Native model import: turn a neutral state or a model file into a fitted
//! sciencekit model that can be inspected, composed and — where the algorithm
//! allows it — trained further (spec `native-model-import`).
//!
//! This is a **pure dispatcher** module: `mod.rs` only declares submodules and
//! re-exports their public items; every implementation lives in its own file.

mod conversion;
mod coverage;
mod descriptor;
mod error;
mod format_detection;
mod loader;
mod model;
mod prediction;
mod retraining;
pub use conversion::{SKNativeImport, sk_import_native_state};
pub use coverage::{SKNativeImportCoverage, sk_native_import_coverage, sk_native_import_supports};
pub use descriptor::{SKNativeModelDescriptor, SKRetrainingSupport};
pub use error::SKNativeImportError;
pub use format_detection::{SKNativeModelFormat, sk_detect_native_model_format};
pub use loader::{sk_import_native_model_from_bytes, sk_import_native_model_from_path};
pub use model::{
    SKNativeClassifier, SKNativeClusterer, SKNativeModelKind, SKNativeRegressor,
    SKNativeTransformer,
};

#[cfg(test)]
mod native_import_tests;
#[cfg(test)]
mod native_retrain_tests;
#[cfg(test)]
mod native_safetensors_tests;
#[cfg(test)]
mod native_validation_tests;
#[cfg(test)]
mod test_models;
