//! Loading a model file into a native model
//! (spec `native-model-import`, requirement "Validation and concurrency").

use sciencekit_common::SKFloat;
use sciencekit_common::builders::SKBuilder;

use super::conversion::{SKNativeImport, sk_import_native_state};
use super::error::SKNativeImportError;
use super::format_detection::{SKNativeModelFormat, sk_detect_native_model_format};
use crate::json::SKJsonDebugReader;
use crate::safetensors::SKSafetensorsReader;

/// Read model bytes (Safetensors or JSON debug) and convert them into a native model.
///
/// The container is detected from the leading bytes, never from a file
/// extension, and pre-validated before any tensor payload is read. A conversion
/// that fails at any stage leaves no model behind.
pub fn sk_import_native_model_from_bytes<F: SKFloat>(
    bytes: &[u8],
) -> Result<SKNativeImport<F>, SKNativeImportError> {
    let state = match sk_detect_native_model_format(bytes)? {
        SKNativeModelFormat::Safetensors => {
            let reader = SKSafetensorsReader::builder().build()?;
            reader.read_state_from_bytes::<F>(bytes)?
        }
        SKNativeModelFormat::Json => {
            let text = std::str::from_utf8(bytes).map_err(|error| {
                SKNativeImportError::Central(sciencekit_common::SKError::Conversion(format!(
                    "the JSON debug model is not valid UTF-8: {error}"
                )))
            })?;
            let reader = SKJsonDebugReader::builder().build()?;
            reader.read_state_from_string::<F>(text)?
        }
    };
    sk_import_native_state(&state)
}

/// Read a model file (Safetensors or JSON debug) and convert it into a native model.
pub fn sk_import_native_model_from_path<F: SKFloat>(
    path: &std::path::Path,
) -> Result<SKNativeImport<F>, SKNativeImportError> {
    let bytes = std::fs::read(path).map_err(sciencekit_common::SKError::from)?;
    sk_import_native_model_from_bytes(&bytes)
}
