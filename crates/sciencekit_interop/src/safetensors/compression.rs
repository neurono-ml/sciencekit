//! Feature-gated Gzip, Brotli and Snappy whole-file codecs (spec `safetensors-format`).

use std::path::Path;

use sciencekit_common::SKError;

#[cfg(feature = "compression-brotli")]
use super::compression_types::SK_BROTLI_MAGIC;
pub(crate) use super::compression_types::SKSafetensorsCompression;
use super::compression_types::sk_detect_safetensors_compression;

/// Compress raw Safetensors `bytes` using `compression`, or fail if its feature is disabled.
pub(crate) fn compress_safetensors_bytes(
    bytes: &[u8],
    compression: SKSafetensorsCompression,
) -> Result<Vec<u8>, SKError> {
    match compression {
        SKSafetensorsCompression::None => Ok(bytes.to_vec()),
        SKSafetensorsCompression::Gzip => compress_gzip(bytes),
        SKSafetensorsCompression::Brotli => compress_brotli(bytes),
        SKSafetensorsCompression::Snappy => compress_snappy(bytes),
    }
}

/// Transparently decompress `bytes` when magic bytes or `path_hint` indicate compression.
pub(crate) fn decompress_safetensors_bytes(
    path_hint: Option<&Path>,
    bytes: &[u8],
) -> Result<Vec<u8>, SKError> {
    let detected = sk_detect_safetensors_compression(bytes)
        .or_else(|| path_hint.and_then(SKSafetensorsCompression::from_path))
        .unwrap_or(SKSafetensorsCompression::None);
    match detected {
        SKSafetensorsCompression::None => Ok(bytes.to_vec()),
        SKSafetensorsCompression::Gzip => decompress_gzip(bytes),
        SKSafetensorsCompression::Brotli => decompress_brotli(bytes),
        SKSafetensorsCompression::Snappy => decompress_snappy(bytes),
    }
}

#[cfg(feature = "compression-gzip")]
fn compress_gzip(bytes: &[u8]) -> Result<Vec<u8>, SKError> {
    use std::io::Write;
    let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    encoder.write_all(bytes)?;
    Ok(encoder.finish()?)
}

#[cfg(not(feature = "compression-gzip"))]
fn compress_gzip(_bytes: &[u8]) -> Result<Vec<u8>, SKError> {
    Err(SKError::UnsupportedOperation {
        operation: "compress_safetensors_gzip",
        reason: "enable Cargo feature `compression-gzip` on `sciencekit_interop`".to_string(),
    })
}

#[cfg(feature = "compression-gzip")]
fn decompress_gzip(bytes: &[u8]) -> Result<Vec<u8>, SKError> {
    use std::io::Read;
    let mut output = Vec::new();
    flate2::read::GzDecoder::new(bytes)
        .read_to_end(&mut output)
        .map_err(|error| {
            SKError::Conversion(format!("corrupt gzip Safetensors payload: {error}"))
        })?;
    Ok(output)
}

#[cfg(not(feature = "compression-gzip"))]
fn decompress_gzip(_bytes: &[u8]) -> Result<Vec<u8>, SKError> {
    Err(SKError::UnsupportedOperation {
        operation: "decompress_safetensors_gzip",
        reason: "enable Cargo feature `compression-gzip` on `sciencekit_interop` to read `.safetensors.gz` files".to_string(),
    })
}

#[cfg(feature = "compression-brotli")]
fn compress_brotli(bytes: &[u8]) -> Result<Vec<u8>, SKError> {
    use std::io::Write;
    let mut output = Vec::from(SK_BROTLI_MAGIC);
    {
        let mut encoder = brotli::CompressorWriter::new(&mut output, 4096, 6, 22);
        encoder.write_all(bytes)?;
        encoder.flush()?;
    }
    Ok(output)
}

#[cfg(not(feature = "compression-brotli"))]
fn compress_brotli(_bytes: &[u8]) -> Result<Vec<u8>, SKError> {
    Err(SKError::UnsupportedOperation {
        operation: "compress_safetensors_brotli",
        reason: "enable Cargo feature `compression-brotli` on `sciencekit_interop`".to_string(),
    })
}

#[cfg(feature = "compression-brotli")]
fn decompress_brotli(bytes: &[u8]) -> Result<Vec<u8>, SKError> {
    use std::io::Read;
    let stream = bytes.strip_prefix(&SK_BROTLI_MAGIC).unwrap_or(bytes);
    let mut output = Vec::new();
    brotli::Decompressor::new(stream, 4096)
        .read_to_end(&mut output)
        .map_err(|error| {
            SKError::Conversion(format!("corrupt Brotli Safetensors payload: {error}"))
        })?;
    Ok(output)
}

#[cfg(not(feature = "compression-brotli"))]
fn decompress_brotli(_bytes: &[u8]) -> Result<Vec<u8>, SKError> {
    Err(SKError::UnsupportedOperation {
        operation: "decompress_safetensors_brotli",
        reason: "enable Cargo feature `compression-brotli` on `sciencekit_interop` to read `.safetensors.brotli` files".to_string(),
    })
}

#[cfg(feature = "compression-snappy")]
fn compress_snappy(bytes: &[u8]) -> Result<Vec<u8>, SKError> {
    use std::io::Write;
    let mut encoder = snap::write::FrameEncoder::new(Vec::new());
    encoder.write_all(bytes)?;
    encoder
        .into_inner()
        .map_err(|error| SKError::Conversion(format!("Snappy flush error: {error}")))
}

#[cfg(not(feature = "compression-snappy"))]
fn compress_snappy(_bytes: &[u8]) -> Result<Vec<u8>, SKError> {
    Err(SKError::UnsupportedOperation {
        operation: "compress_safetensors_snappy",
        reason: "enable Cargo feature `compression-snappy` on `sciencekit_interop`".to_string(),
    })
}

#[cfg(feature = "compression-snappy")]
fn decompress_snappy(bytes: &[u8]) -> Result<Vec<u8>, SKError> {
    use std::io::Read;
    let mut output = Vec::new();
    snap::read::FrameDecoder::new(bytes)
        .read_to_end(&mut output)
        .map_err(|error| {
            SKError::Conversion(format!("corrupt Snappy Safetensors payload: {error}"))
        })?;
    Ok(output)
}

#[cfg(not(feature = "compression-snappy"))]
fn decompress_snappy(_bytes: &[u8]) -> Result<Vec<u8>, SKError> {
    Err(SKError::UnsupportedOperation {
        operation: "decompress_safetensors_snappy",
        reason: "enable Cargo feature `compression-snappy` on `sciencekit_interop` to read `.snappy.safetensors` files".to_string(),
    })
}
