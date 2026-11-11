//! Compression mode enum and magic-byte detection (spec `safetensors-format`).

use std::fmt;
use std::path::Path;

/// Standard RFC 1952 two-byte Gzip stream prefix.
pub const SK_GZIP_MAGIC: [u8; 2] = [0x1F, 0x8B];

/// Four-byte Brotli framing prefix (`"βρ"` in UTF-8) identifying Brotli Safetensors payloads.
pub const SK_BROTLI_MAGIC: [u8; 4] = [0xCE, 0xB2, 0xCF, 0x81];

/// Standard ten-byte Snappy framing stream identifier (`\xff\x06\x00\x00sNaPpY`).
pub const SK_SNAPPY_MAGIC: [u8; 10] = [0xFF, 0x06, 0x00, 0x00, b's', b'N', b'a', b'P', b'p', b'Y'];

/// Whole-file compression variant applied to extended Safetensors payloads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SKSafetensorsCompression {
    /// Uncompressed standard `.safetensors` binary output.
    #[default]
    None,
    /// RFC 1952 Gzip wrapper (`.safetensors.gz`, feature `compression-gzip`).
    Gzip,
    /// Brotli wrapper (`.safetensors.brotli`, feature `compression-brotli`).
    Brotli,
    /// Snappy framed wrapper (`.snappy.safetensors`, feature `compression-snappy`).
    Snappy,
}

impl SKSafetensorsCompression {
    /// Canonical file extension (without leading dot) per PRD §8.4.
    pub fn file_extension(self) -> &'static str {
        match self {
            Self::None => "safetensors",
            Self::Gzip => "safetensors.gz",
            Self::Brotli => "safetensors.brotli",
            Self::Snappy => "snappy.safetensors",
        }
    }

    /// Cargo feature required to encode or decode this variant, if any.
    pub fn feature_flag(self) -> Option<&'static str> {
        match self {
            Self::None => None,
            Self::Gzip => Some("compression-gzip"),
            Self::Brotli => Some("compression-brotli"),
            Self::Snappy => Some("compression-snappy"),
        }
    }

    /// Infer compression from a file path suffix when magic bytes are absent.
    pub fn from_path(path: &Path) -> Option<Self> {
        let name = path.file_name()?.to_string_lossy();
        if name.ends_with(".safetensors.gz") || name.ends_with(".gz") {
            Some(Self::Gzip)
        } else if name.ends_with(".safetensors.brotli")
            || name.ends_with(".brotli")
            || name.ends_with(".br")
        {
            Some(Self::Brotli)
        } else if name.ends_with(".snappy.safetensors") || name.ends_with(".sz") {
            Some(Self::Snappy)
        } else {
            None
        }
    }
}

impl fmt::Display for SKSafetensorsCompression {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => formatter.write_str("none"),
            Self::Gzip => formatter.write_str("gzip"),
            Self::Brotli => formatter.write_str("brotli"),
            Self::Snappy => formatter.write_str("snappy"),
        }
    }
}

/// Detect a compressed Safetensors wrapper purely from leading magic bytes.
pub fn sk_detect_safetensors_compression(bytes: &[u8]) -> Option<SKSafetensorsCompression> {
    if bytes.starts_with(&SK_SNAPPY_MAGIC) {
        Some(SKSafetensorsCompression::Snappy)
    } else if bytes.starts_with(&SK_BROTLI_MAGIC) {
        Some(SKSafetensorsCompression::Brotli)
    } else if bytes.starts_with(&SK_GZIP_MAGIC) {
        Some(SKSafetensorsCompression::Gzip)
    } else {
        None
    }
}
