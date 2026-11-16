//! Model artifact adapters over the interop Safetensors and JSON formats and
//! the optional ONNX translator.

use std::path::Path;

use sciencekit::SKModelState;
use sciencekit::interop::json::{SKJsonFormatMode, sk_write_model_state_json_file};
use sciencekit::interop::native::SKNativeImport;
use sciencekit::interop::safetensors::sk_write_model_state_safetensors_file;

use crate::error::SKCliError;

/// The model artifact formats the CLI reads and writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKModelFormat {
    /// The extended Safetensors internal format.
    Safetensors,
    /// The human-readable JSON debug format.
    Json,
    /// ONNX (export and import behind the `onnx` feature).
    Onnx,
}

impl SKModelFormat {
    /// Parse a configured or command-line format name.
    pub fn from_name(name: &str) -> Result<Self, SKCliError> {
        match name.to_ascii_lowercase().as_str() {
            "safetensors" | "safetensor" => Ok(SKModelFormat::Safetensors),
            "json" => Ok(SKModelFormat::Json),
            "onnx" => Ok(SKModelFormat::Onnx),
            other => Err(SKCliError::validation(
                "configuration.model_format",
                format!("`{other}` is not a model format; use `safetensors`, `json` or `onnx`"),
            )),
        }
    }

    /// Infer the format from the extension, defaulting to Safetensors.
    pub fn from_path(path: &Path) -> Self {
        match path
            .extension()
            .and_then(|extension| extension.to_str())
            .map(str::to_ascii_lowercase)
            .as_deref()
        {
            Some("json") => SKModelFormat::Json,
            Some("onnx") => SKModelFormat::Onnx,
            _ => SKModelFormat::Safetensors,
        }
    }

    /// The stable format name used in diagnostics.
    pub const fn name(self) -> &'static str {
        match self {
            SKModelFormat::Safetensors => "safetensors",
            SKModelFormat::Json => "json",
            SKModelFormat::Onnx => "onnx",
        }
    }
}

/// Write a model state in the requested format.
pub fn sk_save_state(
    path: &Path,
    state: &SKModelState<f64>,
    format: SKModelFormat,
) -> Result<(), SKCliError> {
    let text = path.display().to_string();
    match format {
        SKModelFormat::Safetensors => sk_write_model_state_safetensors_file(path, state)
            .map_err(|error| SKCliError::data(&text, error.to_string())),
        SKModelFormat::Json => {
            sk_write_model_state_json_file(path, state, SKJsonFormatMode::Pretty)
                .map_err(|error| SKCliError::data(&text, error.to_string()))
        }
        SKModelFormat::Onnx => {
            #[cfg(feature = "onnx")]
            {
                let model = sciencekit::interop::sk_export_state_to_onnx(state)
                    .map_err(|error| SKCliError::runtime(format!("ONNX export: {error}")))?;
                model
                    .save(path)
                    .map_err(|error| SKCliError::data(&text, error.to_string()))
            }
            #[cfg(not(feature = "onnx"))]
            {
                Err(SKCliError::unsupported("onnx", "an ONNX model artifact"))
            }
        }
    }
}

/// Read a model state from a Safetensors or JSON artifact.
pub fn sk_load_state(path: &Path) -> Result<SKModelState<f64>, SKCliError> {
    let text = path.display().to_string();
    match SKModelFormat::from_path(path) {
        SKModelFormat::Json => sciencekit::interop::json::sk_read_model_state_json_file(path)
            .map_err(|error| SKCliError::data(&text, error.to_string())),
        SKModelFormat::Safetensors => {
            sciencekit::interop::safetensors::sk_read_model_state_safetensors_file(path)
                .map_err(|error| SKCliError::data(&text, error.to_string()))
        }
        SKModelFormat::Onnx => Err(SKCliError::unsupported(
            "onnx",
            "reading an ONNX model state; load it with predict instead",
        )),
    }
}

/// Import a model artifact into a native model that can predict or transform.
pub fn sk_load_native(path: &Path) -> Result<SKNativeImport<f64>, SKCliError> {
    let text = path.display().to_string();
    match SKModelFormat::from_path(path) {
        SKModelFormat::Onnx => {
            #[cfg(feature = "onnx")]
            {
                sciencekit::interop::SKOnnxNativeImporter::builder()
                    .model_file(path)
                    .build::<f64>()
                    .map_err(|error| SKCliError::data(&text, error.to_string()))
            }
            #[cfg(not(feature = "onnx"))]
            {
                Err(SKCliError::unsupported("onnx", "an ONNX model artifact"))
            }
        }
        _ => sciencekit::interop::native::sk_import_native_model_from_path(path)
            .map_err(|error| SKCliError::data(&text, error.to_string())),
    }
}

/// Build the inspection document describing a model artifact.
pub fn sk_inspect(path: &Path) -> Result<serde_json::Value, SKCliError> {
    let state = sk_load_state(path)?;
    let metadata = state.metadata();
    Ok(serde_json::json!({
        "path": path.display().to_string(),
        "format": SKModelFormat::from_path(path).name(),
        "estimator": metadata.estimator_name(),
        "format_version": metadata.format_version(),
        "feature_count": metadata.feature_count(),
        "tensor_count": state.number_of_tensors(),
        "parameter_count": metadata.parameters().len(),
        "tensors": state.tensors().keys().cloned().collect::<Vec<_>>(),
    }))
}
