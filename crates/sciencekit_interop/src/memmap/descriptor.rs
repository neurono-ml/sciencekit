//! The sidecar descriptor of a raw binary matrix (spec `memory-mapped-sources`).
//!
//! A raw file carries no shape or dtype, so the source reads them from an
//! explicit descriptor or a sidecar JSON file:
//!
//! ```json
//! {"rows": 1024, "columns": 8, "dtype": "f64"}
//! ```

use std::path::Path;

use sciencekit_common::SKError;
use serde::{Deserialize, Serialize};

/// The scalar dtype a raw binary file stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SKRawBinaryDtype {
    /// IEEE-754 single precision, four bytes per value.
    #[serde(rename = "f32")]
    Float32,
    /// IEEE-754 double precision, eight bytes per value.
    #[serde(rename = "f64")]
    Float64,
    /// Two's-complement signed 64-bit integer.
    #[serde(rename = "i64")]
    Int64,
}

impl SKRawBinaryDtype {
    /// The canonical name used in descriptors and errors.
    pub fn name(self) -> &'static str {
        match self {
            SKRawBinaryDtype::Float32 => "f32",
            SKRawBinaryDtype::Float64 => "f64",
            SKRawBinaryDtype::Int64 => "i64",
        }
    }

    /// The size of one value in bytes.
    pub fn element_size(self) -> usize {
        match self {
            SKRawBinaryDtype::Float32 => 4,
            SKRawBinaryDtype::Float64 | SKRawBinaryDtype::Int64 => 8,
        }
    }
}

/// The byte order a raw binary file declares.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SKRawBinaryEndianness {
    /// Little-endian, the only byte order the adapter reads zero-copy.
    #[default]
    Little,
    /// Big-endian, rejected at open time.
    Big,
}

/// Shape and dtype of a raw binary matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct SKRawBinaryDescriptor {
    /// The number of data rows.
    pub rows: usize,
    /// The number of columns.
    pub columns: usize,
    /// The scalar dtype.
    pub dtype: SKRawBinaryDtype,
    /// The byte order; defaults to little-endian.
    #[serde(default)]
    pub endianness: SKRawBinaryEndianness,
}

impl SKRawBinaryDescriptor {
    /// Build a little-endian descriptor explicitly.
    pub fn new(rows: usize, columns: usize, dtype: SKRawBinaryDtype) -> Self {
        SKRawBinaryDescriptor {
            rows,
            columns,
            dtype,
            endianness: SKRawBinaryEndianness::Little,
        }
    }

    /// Parse a descriptor from JSON text.
    ///
    /// # Errors
    /// Returns [`SKError::MalformedMappedHeader`] naming the descriptor when the
    /// JSON is invalid or omits a required field.
    pub fn from_json(text: &str, source_name: &str) -> Result<Self, SKError> {
        serde_json::from_str(text).map_err(|error| SKError::MalformedMappedHeader {
            path: source_name.to_string(),
            format: "raw-binary-descriptor",
            reason: error.to_string(),
        })
    }

    /// Read a sidecar JSON descriptor from disk.
    ///
    /// # Errors
    /// Returns the central I/O error when the sidecar is unreadable and
    /// [`SKError::MalformedMappedHeader`] when its JSON is invalid.
    pub fn from_sidecar(path: &Path) -> Result<Self, SKError> {
        let text = std::fs::read_to_string(path)?;
        Self::from_json(&text, &path.display().to_string())
    }

    /// Serialize the descriptor back to JSON.
    pub fn to_json(&self) -> String {
        serde_json::to_string(self).expect("a descriptor always serializes")
    }

    /// The number of data rows.
    pub fn number_of_rows(&self) -> usize {
        self.rows
    }

    /// The number of columns.
    pub fn number_of_columns(&self) -> usize {
        self.columns
    }

    /// The scalar dtype.
    pub fn dtype(&self) -> SKRawBinaryDtype {
        self.dtype
    }

    /// Whether the descriptor declares little-endian data.
    pub fn is_little_endian(&self) -> bool {
        self.endianness == SKRawBinaryEndianness::Little
    }

    /// The byte length the declared shape requires.
    ///
    /// # Errors
    /// Returns [`SKError::MalformedMappedHeader`] when the multiplication
    /// overflows the platform's `usize`.
    pub fn required_byte_length(&self) -> Result<u64, SKError> {
        let elements = self
            .rows
            .checked_mul(self.columns)
            .and_then(|count| count.checked_mul(self.dtype.element_size()))
            .ok_or_else(|| SKError::MalformedMappedHeader {
                path: format!(
                    "raw-binary-descriptor ({} rows, {} columns)",
                    self.rows, self.columns
                ),
                format: "raw-binary-descriptor",
                reason: "the declared shape overflows the platform's address space".to_string(),
            })?;
        Ok(elements as u64)
    }
}
