//! Container detection and pre-validation for model files
//! (spec `native-model-import`, requirement "Validation and concurrency").
//!
//! Format detection is the first act of the trust boundary. Safetensors and the
//! JSON debug format are both JSON-shaped, so the discriminator is structural:
//! a JSON document starts with `{`, a Safetensors file starts with an
//! eight-byte little-endian header length. Compressed Safetensors keep their
//! compression magic and are still recognized as Safetensors.
//!
//! The header is parsed once here, before any tensor payload is touched, so a
//! wrong magic, a truncated file and an unknown format version are reported with
//! their own error variants instead of surfacing as a generic decode failure.

use sciencekit_common::SKError;

use super::error::SKNativeImportError;
use crate::json::SK_JSON_DEBUG_FORMAT_VERSION;
use crate::safetensors::{
    SK_BROTLI_MAGIC, SK_GZIP_MAGIC, SK_SAFETENSORS_FORMAT_VERSION, SK_SNAPPY_MAGIC,
    sk_detect_safetensors_compression,
};

/// The container families the native importer can read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKNativeModelFormat {
    /// The extended binary Safetensors model format.
    Safetensors,
    /// The human-readable JSON debug representation.
    Json,
}

impl SKNativeModelFormat {
    /// The canonical container name used in diagnostics.
    pub fn name(self) -> &'static str {
        match self {
            Self::Safetensors => "safetensors",
            Self::Json => "json",
        }
    }
}

/// Detect the container family from the leading bytes.
///
/// Returns [`SKNativeImportError::WrongMagic`] when the bytes are neither a JSON
/// document nor a Safetensors file (raw or compressed), and validates the
/// declared format version before returning.
pub fn sk_detect_native_model_format(
    bytes: &[u8],
) -> Result<SKNativeModelFormat, SKNativeImportError> {
    if is_compressed_safetensors(bytes) {
        return Ok(SKNativeModelFormat::Safetensors);
    }
    if first_non_whitespace(bytes) == Some(b'{') {
        validate_json_version(bytes)?;
        return Ok(SKNativeModelFormat::Json);
    }
    validate_safetensors_header(bytes)?;
    Ok(SKNativeModelFormat::Safetensors)
}

/// The first byte that is not JSON whitespace.
fn first_non_whitespace(bytes: &[u8]) -> Option<u8> {
    bytes
        .iter()
        .copied()
        .find(|byte| !matches!(byte, b' ' | b'\n' | b'\r' | b'\t'))
}

fn is_compressed_safetensors(bytes: &[u8]) -> bool {
    bytes.starts_with(&SK_GZIP_MAGIC)
        || bytes.starts_with(&SK_BROTLI_MAGIC)
        || bytes.starts_with(&SK_SNAPPY_MAGIC)
        || sk_detect_safetensors_compression(bytes).is_some()
}

fn wrong_magic(bytes: &[u8], expected: &'static str) -> SKNativeImportError {
    SKNativeImportError::WrongMagic {
        expected,
        found: bytes.iter().copied().take(8).collect(),
    }
}

fn validate_safetensors_header(bytes: &[u8]) -> Result<(), SKNativeImportError> {
    let Some(length_bytes) = bytes.get(..8) else {
        return Err(wrong_magic(
            bytes,
            "an eight-byte Safetensors header length",
        ));
    };
    let header_length = u64::from_le_bytes(
        length_bytes
            .try_into()
            .map_err(|_| wrong_magic(bytes, "an eight-byte Safetensors header length"))?,
    );
    let available = (bytes.len() - 8) as u64;
    if header_length > available {
        // `header_length` comes from untrusted bytes: adding the eight header
        // length bytes must not overflow, so saturate the reported total.
        return Err(SKNativeImportError::TruncatedFile {
            format: "safetensors",
            needed: header_length.saturating_add(8),
            available: bytes.len() as u64,
        });
    }
    let header = bytes
        .get(8..8 + header_length as usize)
        .ok_or_else(|| wrong_magic(bytes, "a complete Safetensors header"))?;
    let parsed: serde_json::Value = serde_json::from_slice(header)
        .map_err(|_| wrong_magic(bytes, "a JSON Safetensors header"))?;
    let declared = parsed
        .get("__metadata__")
        .and_then(|metadata| metadata.get("sciencekit.format_version"))
        .and_then(serde_json::Value::as_str)
        .and_then(|raw| raw.split('.').next())
        .and_then(|major| major.parse::<u64>().ok());
    match declared {
        Some(version) if version != u64::from(SK_SAFETENSORS_FORMAT_VERSION) => {
            Err(SKNativeImportError::UnknownFormatVersion {
                format: "safetensors",
                found: version,
                supported: u64::from(SK_SAFETENSORS_FORMAT_VERSION),
            })
        }
        Some(_) => Ok(()),
        None => Err(SKNativeImportError::UnsupportedConstruct {
            construct: "sciencekit.format_version".to_string(),
            reason: "the Safetensors header does not declare a sciencekit model format version"
                .to_string(),
        }),
    }
}

fn validate_json_version(bytes: &[u8]) -> Result<(), SKNativeImportError> {
    let parsed: serde_json::Value = serde_json::from_slice(bytes).map_err(|error| {
        SKNativeImportError::Central(SKError::Conversion(format!(
            "the JSON debug model is not valid JSON: {error}"
        )))
    })?;
    let declared = parsed
        .get("format_version")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| SKNativeImportError::UnsupportedConstruct {
            construct: "format_version".to_string(),
            reason: "the JSON document does not declare a container format version".to_string(),
        })?;
    if declared != u64::from(SK_JSON_DEBUG_FORMAT_VERSION) {
        return Err(SKNativeImportError::UnknownFormatVersion {
            format: "json",
            found: declared,
            supported: u64::from(SK_JSON_DEBUG_FORMAT_VERSION),
        });
    }
    Ok(())
}
