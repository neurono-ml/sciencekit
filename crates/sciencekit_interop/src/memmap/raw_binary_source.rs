//! Random-access source over a raw binary matrix (spec `memory-mapped-sources`).

use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use memmap2::Mmap;
use ndarray::ArrayView1;
use sciencekit_common::{SKDataBatch, SKError, SKFloat, SKLazySource, SKMappableSource};

use super::alignment::sk_scalar_slice;
use super::descriptor::SKRawBinaryDescriptor;
use super::mapping::{sk_map_read_only, sk_open_file};
use super::sequential::sk_next_mapped_block;

/// A memory-mapped raw binary matrix with O(1) row access.
///
/// The file stores `rows × columns` fixed-width scalars in C order with no
/// header; the shape and dtype come from a [`SKRawBinaryDescriptor`]. Rows are
/// borrowed directly from the mapping, so access never copies and never reads
/// preceding rows.
#[derive(Debug)]
pub struct SKRawBinaryMappableSource<F> {
    path: PathBuf,
    mapping: Arc<Mmap>,
    number_of_rows: usize,
    number_of_columns: usize,
    block_size: usize,
    cursor: AtomicUsize,
    marker: PhantomData<F>,
}

impl<F: SKFloat> SKRawBinaryMappableSource<F> {
    /// Open and map a raw binary file against its descriptor.
    ///
    /// The declared shape must match the file length exactly, the declared
    /// dtype must match the model scalar, and the data must be little-endian.
    ///
    /// # Errors
    /// Returns a structured error for a missing file, denied permission, a
    /// non-little-endian declaration, a dtype the scalar cannot view, or a
    /// length that differs from the declared shape.
    pub fn open(
        path: impl AsRef<Path>,
        descriptor: SKRawBinaryDescriptor,
    ) -> Result<Self, SKError> {
        let path = path.as_ref();
        if !descriptor.is_little_endian() {
            return Err(SKError::NonLittleEndianMappedData {
                path: path.display().to_string(),
                endianness: "big".to_string(),
            });
        }
        if descriptor.dtype().name() != F::SCALAR_NAME {
            return Err(SKError::UnsupportedMappedDtype {
                path: path.display().to_string(),
                dtype: descriptor.dtype().name().to_string(),
                supported: F::SCALAR_NAME.to_string(),
            });
        }
        let expected_bytes = descriptor.required_byte_length()?;
        let file = sk_open_file(path)?;
        let actual_bytes = file.metadata()?.len();
        if actual_bytes != expected_bytes {
            return Err(SKError::MappedLengthMismatch {
                path: path.display().to_string(),
                expected_bytes,
                actual_bytes,
            });
        }
        let mapping = Arc::new(sk_map_read_only(&file, path)?);
        // Constructing the slice checks alignment once, at open time, so every
        // later row view is guaranteed to be aligned.
        sk_scalar_slice::<F>(&mapping, path)?;
        Ok(SKRawBinaryMappableSource {
            path: path.to_path_buf(),
            mapping,
            number_of_rows: descriptor.number_of_rows(),
            number_of_columns: descriptor.number_of_columns(),
            block_size: 1024,
            cursor: AtomicUsize::new(0),
            marker: PhantomData,
        })
    }

    /// Set the maximum number of rows per sequential batch.
    ///
    /// # Errors
    /// Rejects a zero block size.
    pub fn with_block_size(mut self, block_size: usize) -> Result<Self, SKError> {
        if block_size == 0 {
            return Err(SKError::InvalidHyperparameter {
                name: "block_size",
                reason: "a streaming block must contain at least one row".to_string(),
            });
        }
        self.block_size = block_size;
        Ok(self)
    }

    /// The number of columns.
    pub fn number_of_columns(&self) -> usize {
        self.number_of_columns
    }

    /// The file this source maps.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The address of the first payload byte, for pointer-identity checks.
    pub fn mapped_payload_address(&self) -> usize {
        self.mapping.as_ptr() as usize
    }
}

impl<F: SKFloat> SKMappableSource<F> for SKRawBinaryMappableSource<F> {
    type Error = SKError;

    fn number_of_rows(&self) -> usize {
        self.number_of_rows
    }

    fn row(&self, index: usize) -> Result<ArrayView1<'_, F>, Self::Error> {
        if index >= self.number_of_rows {
            return Err(SKError::Conversion(format!(
                "row {index} is out of bounds for `{}` with {} rows",
                self.path.display(),
                self.number_of_rows
            )));
        }
        // SAFETY: alignment was validated at open time and the length check
        // guarantees this many elements exist.
        let values = sk_scalar_slice::<F>(&self.mapping, &self.path)?;
        let start = index * self.number_of_columns;
        let row = &values[start..start + self.number_of_columns];
        Ok(ArrayView1::from_shape(self.number_of_columns, row)
            .expect("a row slice always reshapes to a one-dimensional view"))
    }
}

impl<F: SKFloat> SKLazySource<F> for SKRawBinaryMappableSource<F> {
    type Error = SKError;

    fn next_batch(&mut self) -> Result<Option<SKDataBatch<F>>, Self::Error> {
        sk_next_mapped_block(self, &self.cursor, self.block_size)
    }
}
