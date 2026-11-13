//! Random-access and sequential source over a NumPy `.npy` matrix (spec
//! `memory-mapped-sources`).

use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use memmap2::Mmap;
use ndarray::{Array2, ArrayView1};
use sciencekit_common::{SKDataBatch, SKError, SKFloat, SKLazySource, SKMappableSource};

use super::alignment::sk_scalar_slice;
use super::mapping::{sk_map_read_only, sk_open_file};
use super::numpy_header::{SKNumpyDtype, SKNumpyHeader, sk_parse_numpy_header};
use super::sequential::sk_next_mapped_block;

/// The storage behind a mapped `.npy` source.
#[derive(Debug)]
enum SKNumpyBacking<F> {
    /// A zero-copy view into the memory mapping.
    Mapped {
        mapping: Arc<Mmap>,
        data_offset: usize,
    },
    /// An owned matrix produced by the explicit casting path.
    Materialized(Array2<F>),
}

/// A memory-mapped C-order NumPy matrix with O(1) row access.
///
/// A dtype that matches the model scalar (`f4` for `f32`, `f8` for `f64`) is
/// viewed in place; any other supported dtype requires
/// [`Self::open_with_cast`], which materializes the matrix once.
#[derive(Debug)]
pub struct SKNumpyMappableSource<F> {
    path: PathBuf,
    header: SKNumpyHeader,
    backing: SKNumpyBacking<F>,
    block_size: usize,
    cursor: AtomicUsize,
    marker: PhantomData<F>,
}

impl<F: SKFloat> SKNumpyMappableSource<F> {
    /// Open a `.npy` array, requiring a dtype that matches the model scalar.
    ///
    /// The file must not be truncated or rewritten while the source is alive;
    /// see the module-level file-stability contract for the failure mode.
    ///
    /// # Errors
    /// Returns a structured error for a missing file, a malformed or truncated
    /// header, Fortran order, a non-little-endian dtype, a shape whose byte
    /// length differs from the file, a misaligned payload, or a dtype that is
    /// not the model scalar (use [`Self::open_with_cast`] for that).
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SKError> {
        Self::open_internal(path.as_ref(), false)
    }

    /// Open a `.npy` array and copy-convert its values to the model scalar.
    ///
    /// This is the documented path for `int64` (and for any float dtype that
    /// differs from the model scalar): it materializes the matrix in memory.
    ///
    /// The file must not be truncated or rewritten while it is being mapped and
    /// converted; see the module-level file-stability contract.
    ///
    /// # Errors
    /// Returns the same structured opening errors as [`Self::open`].
    pub fn open_with_cast(path: impl AsRef<Path>) -> Result<Self, SKError> {
        Self::open_internal(path.as_ref(), true)
    }

    fn open_internal(path: &Path, cast: bool) -> Result<Self, SKError> {
        let file = sk_open_file(path)?;
        let mapping = Arc::new(sk_map_read_only(&file, path)?);
        let path_name = path.display().to_string();
        let header = sk_parse_numpy_header(&mapping, &path_name)?;
        let element_count = header
            .number_of_rows
            .checked_mul(header.number_of_columns)
            .ok_or_else(|| SKError::MalformedMappedHeader {
                path: path_name.clone(),
                format: "npy",
                reason: "the declared shape overflows the platform's address space".to_string(),
            })?;
        // A crafted header can declare a shape whose element count fits `usize`
        // while its byte length does not; the multiplication must be checked so
        // a hostile file becomes a structured error instead of an overflow.
        let expected_bytes = u64::try_from(element_count)
            .ok()
            .and_then(|count| count.checked_mul(header.dtype.element_size() as u64))
            .ok_or_else(|| SKError::MalformedMappedHeader {
                path: path_name.clone(),
                format: "npy",
                reason: "the declared shape overflows the platform's address space".to_string(),
            })?;
        let actual_bytes = (mapping.len() - header.data_offset) as u64;
        if actual_bytes != expected_bytes {
            return Err(SKError::MappedLengthMismatch {
                path: path_name.clone(),
                expected_bytes,
                actual_bytes,
            });
        }
        let matches_scalar = sk_dtype_matches_scalar::<F>(header.dtype);
        let backing = if matches_scalar && !cast {
            // Validate alignment once, at open time, so row views cannot fail.
            sk_scalar_slice::<F>(&mapping[header.data_offset..], path)?;
            SKNumpyBacking::Mapped {
                mapping,
                data_offset: header.data_offset,
            }
        } else {
            if !cast {
                return Err(SKError::UnsupportedMappedDtype {
                    path: path_name,
                    dtype: header.dtype.name().to_string(),
                    supported: format!(
                        "{} (open the file with `SKNumpyMappableSource::open_with_cast` to copy-convert it)",
                        F::SCALAR_NAME
                    ),
                });
            }
            let values = sk_convert_payload::<F>(
                &mapping[header.data_offset..],
                header.dtype,
                element_count,
                path,
            )?;
            SKNumpyBacking::Materialized(
                Array2::from_shape_vec((header.number_of_rows, header.number_of_columns), values)
                    .expect("the payload length matches the declared shape"),
            )
        };
        Ok(SKNumpyMappableSource {
            path: path.to_path_buf(),
            header,
            backing,
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
        self.header.number_of_columns
    }

    /// The dtype the file declares.
    pub fn dtype_name(&self) -> &'static str {
        self.header.dtype.name()
    }

    /// Whether rows alias the mapping instead of an owned copy.
    pub fn is_zero_copy(&self) -> bool {
        matches!(self.backing, SKNumpyBacking::Mapped { .. })
    }

    /// The address of the first payload byte, for pointer-identity checks.
    ///
    /// Returns `None` on the materialized casting path.
    pub fn mapped_payload_address(&self) -> Option<usize> {
        match &self.backing {
            SKNumpyBacking::Mapped {
                mapping,
                data_offset,
            } => Some(mapping.as_ptr() as usize + data_offset),
            SKNumpyBacking::Materialized(_) => None,
        }
    }

    /// The file this source maps.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl<F: SKFloat> SKMappableSource<F> for SKNumpyMappableSource<F> {
    type Error = SKError;

    fn number_of_rows(&self) -> usize {
        self.header.number_of_rows
    }

    fn row(&self, index: usize) -> Result<ArrayView1<'_, F>, Self::Error> {
        if index >= self.header.number_of_rows {
            return Err(SKError::Conversion(format!(
                "row {index} is out of bounds for `{}` with {} rows",
                self.path.display(),
                self.header.number_of_rows
            )));
        }
        match &self.backing {
            SKNumpyBacking::Mapped {
                mapping,
                data_offset,
            } => {
                // SAFETY: the dtype and alignment were validated at open time.
                let values = sk_scalar_slice::<F>(&mapping[*data_offset..], &self.path)?;
                let start = index * self.header.number_of_columns;
                let row = &values[start..start + self.header.number_of_columns];
                Ok(ArrayView1::from_shape(self.header.number_of_columns, row)
                    .expect("a row slice always reshapes to a one-dimensional view"))
            }
            SKNumpyBacking::Materialized(matrix) => Ok(matrix.row(index)),
        }
    }
}

impl<F: SKFloat> SKLazySource<F> for SKNumpyMappableSource<F> {
    type Error = SKError;

    fn next_batch(&mut self) -> Result<Option<SKDataBatch<F>>, Self::Error> {
        sk_next_mapped_block(self, &self.cursor, self.block_size)
    }
}

fn sk_dtype_matches_scalar<F: SKFloat>(dtype: SKNumpyDtype) -> bool {
    matches!(
        (dtype, F::SCALAR_NAME),
        (SKNumpyDtype::Float32, "f32") | (SKNumpyDtype::Float64, "f64")
    )
}

fn sk_convert_payload<F: SKFloat>(
    bytes: &[u8],
    dtype: SKNumpyDtype,
    element_count: usize,
    path: &Path,
) -> Result<Vec<F>, SKError> {
    let element_size = dtype.element_size();
    let mut values = Vec::with_capacity(element_count);
    for chunk in bytes.chunks_exact(element_size) {
        let value = match dtype {
            SKNumpyDtype::Float32 => num_traits::NumCast::from(f32::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3],
            ])),
            SKNumpyDtype::Float64 => num_traits::NumCast::from(f64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])),
            SKNumpyDtype::Int64 => num_traits::NumCast::from(i64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ])),
        };
        values.push(value.ok_or_else(|| {
            SKError::Conversion(format!(
                "a `{}` value in `{}` cannot be represented as `{}`",
                dtype.name(),
                path.display(),
                F::SCALAR_NAME
            ))
        })?);
    }
    Ok(values)
}
