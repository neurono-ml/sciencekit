//! Random-access and sequential source over an Arrow IPC file (spec
//! `memory-mapped-sources`).

use std::io::Cursor;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::AtomicUsize;

use arrow_array::{Array, Float32Array, Float64Array, RecordBatch};
use arrow_ipc::reader::FileReader;
use memmap2::Mmap;
use ndarray::{Array2, ArrayView1};
use sciencekit_common::{SKDataBatch, SKError, SKFloat, SKLazySource, SKMappableSource};

use super::arrow_ipc_layout::{SKArrowIpcLayout, sk_read_ipc_layout};
use super::mapping::{sk_map_read_only, sk_open_file};
use super::sequential::sk_next_mapped_block;

/// A memory-mapped Arrow IPC file with O(1) batch lookup and row access.
///
/// Opening reads only the footer and each batch's message metadata. A row
/// lookup binary-searches the cumulative row offsets and decodes just the batch
/// that contains the row; decoded batches are cached as row-major scalars.
#[derive(Debug)]
pub struct SKArrowIpcMappableSource<F> {
    path: PathBuf,
    mapping: Arc<Mmap>,
    layout: SKArrowIpcLayout,
    block_size: usize,
    cursor: AtomicUsize,
    cache: Vec<OnceLock<Box<[F]>>>,
    marker: PhantomData<F>,
}

impl<F: SKFloat> SKArrowIpcMappableSource<F> {
    /// Open and index an Arrow IPC file without decoding its data batches.
    ///
    /// The file must not be truncated or rewritten while the source is alive;
    /// see the module-level file-stability contract for the failure mode.
    ///
    /// # Errors
    /// Returns a structured error for a missing file, a malformed header, a
    /// truncated batch, or a column dtype other than `Float32`/`Float64`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SKError> {
        let path = path.as_ref();
        let file = sk_open_file(path)?;
        let mapping = Arc::new(sk_map_read_only(&file, path)?);
        let layout = sk_read_ipc_layout(&mapping, &path.display().to_string())?;
        let cache = (0..layout.number_of_batches())
            .map(|_| OnceLock::new())
            .collect();
        Ok(SKArrowIpcMappableSource {
            path: path.to_path_buf(),
            mapping,
            layout,
            block_size: 1024,
            cursor: AtomicUsize::new(0),
            cache,
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

    /// The number of columns in the Arrow schema.
    pub fn number_of_columns(&self) -> usize {
        self.layout.schema.fields().len()
    }

    /// The number of indexed record batches.
    pub fn number_of_batches(&self) -> usize {
        self.layout.number_of_batches()
    }

    /// How many batches have been decoded so far.
    ///
    /// Opening leaves this at zero; a row access decodes only its own batch.
    pub fn decoded_batch_count(&self) -> usize {
        self.cache
            .iter()
            .filter(|entry| entry.get().is_some())
            .count()
    }

    /// The file this source maps.
    pub fn path(&self) -> &Path {
        &self.path
    }

    fn sk_decode_batch(&self, index: usize) -> Result<Box<[F]>, SKError> {
        let number_of_columns = self.number_of_columns();
        let mut reader =
            FileReader::try_new(Cursor::new(&self.mapping[..]), None).map_err(|error| {
                SKError::UnreadableMappedBatch {
                    path: self.path.display().to_string(),
                    batch: index,
                    reason: error.to_string(),
                }
            })?;
        reader
            .set_index(index)
            .map_err(|error| SKError::UnreadableMappedBatch {
                path: self.path.display().to_string(),
                batch: index,
                reason: error.to_string(),
            })?;
        let record_batch = reader
            .next()
            .ok_or_else(|| SKError::UnreadableMappedBatch {
                path: self.path.display().to_string(),
                batch: index,
                reason: "the file ends before the requested batch".to_string(),
            })?
            .map_err(|error| SKError::UnreadableMappedBatch {
                path: self.path.display().to_string(),
                batch: index,
                reason: error.to_string(),
            })?;
        sk_materialize_batch::<F>(
            &record_batch,
            number_of_columns,
            self.layout.batches[index].number_of_rows,
            &self.path,
            index,
        )
    }
}

impl<F: SKFloat> SKMappableSource<F> for SKArrowIpcMappableSource<F> {
    type Error = SKError;

    fn number_of_rows(&self) -> usize {
        self.layout.number_of_rows
    }

    fn row(&self, index: usize) -> Result<ArrayView1<'_, F>, Self::Error> {
        if index >= self.layout.number_of_rows {
            return Err(SKError::Conversion(format!(
                "row {index} is out of bounds for `{}` with {} rows",
                self.path.display(),
                self.layout.number_of_rows
            )));
        }
        let batch_index = self.layout.batch_index_for(index);
        let row_in_batch = index - self.layout.batches[batch_index].first_row;
        let entry = &self.cache[batch_index];
        if entry.get().is_none() {
            let decoded = self.sk_decode_batch(batch_index)?;
            // A concurrent reader may have won the race; either decoded batch
            // is identical, and the losing value is simply dropped.
            let _ = entry.set(decoded);
        }
        let values = entry.get().expect("the batch was decoded above");
        let number_of_columns = self.number_of_columns();
        let number_of_values = values.len();
        let start = row_in_batch
            .checked_mul(number_of_columns)
            .filter(|start| {
                start
                    .checked_add(number_of_columns)
                    .is_some_and(|end| end <= number_of_values)
            })
            .ok_or_else(|| SKError::UnreadableMappedBatch {
                path: self.path.display().to_string(),
                batch: batch_index,
                reason: format!(
                    "row {row_in_batch} of {} does not fit the {number_of_values} decoded values",
                    self.layout.batches[batch_index].number_of_rows
                ),
            })?;
        let end = start + number_of_columns;
        Ok(
            ArrayView1::from_shape(number_of_columns, &values[start..end])
                .expect("a row slice always reshapes to a one-dimensional view"),
        )
    }
}

impl<F: SKFloat> SKLazySource<F> for SKArrowIpcMappableSource<F> {
    type Error = SKError;

    fn next_batch(&mut self) -> Result<Option<SKDataBatch<F>>, Self::Error> {
        sk_next_mapped_block(self, &self.cursor, self.block_size)
    }
}

fn sk_materialize_batch<F: SKFloat>(
    record_batch: &RecordBatch,
    number_of_columns: usize,
    expected_rows: usize,
    path: &Path,
    batch: usize,
) -> Result<Box<[F]>, SKError> {
    let number_of_rows = record_batch.num_rows();
    if number_of_rows != expected_rows {
        return Err(SKError::UnreadableMappedBatch {
            path: path.display().to_string(),
            batch,
            reason: format!(
                "the batch metadata declares {expected_rows} rows but the decoded batch carries {number_of_rows}"
            ),
        });
    }
    let mut matrix = Array2::<F>::zeros((number_of_rows, number_of_columns));
    for (column_index, array) in record_batch.columns().iter().enumerate() {
        let values = sk_column_values::<F>(array, path, batch)?;
        matrix
            .column_mut(column_index)
            .assign(&ArrayView1::from(values.as_slice()));
    }
    let (values, offset) = matrix.into_raw_vec_and_offset();
    debug_assert!(offset.unwrap_or(0) == 0);
    Ok(values.into_boxed_slice())
}

fn sk_column_values<F: SKFloat>(
    array: &arrow_array::ArrayRef,
    path: &Path,
    batch: usize,
) -> Result<Vec<F>, SKError> {
    if let Some(values) = array.as_any().downcast_ref::<Float64Array>() {
        return values
            .iter()
            .map(|value| sk_scalar(value, path, batch, "f64"))
            .collect();
    }
    if let Some(values) = array.as_any().downcast_ref::<Float32Array>() {
        return values
            .iter()
            .map(|value| sk_scalar(value.map(f64::from), path, batch, "f32"))
            .collect();
    }
    Err(SKError::UnsupportedMappedDtype {
        path: path.display().to_string(),
        dtype: array.data_type().to_string(),
        supported: "Float32, Float64".to_string(),
    })
}

fn sk_scalar<F: SKFloat>(
    value: Option<f64>,
    path: &Path,
    batch: usize,
    dtype: &str,
) -> Result<F, SKError> {
    let value = value.ok_or_else(|| SKError::UnreadableMappedBatch {
        path: path.display().to_string(),
        batch,
        reason: format!("a `{dtype}` column carries a null value"),
    })?;
    num_traits::NumCast::from(value).ok_or_else(|| SKError::UnreadableMappedBatch {
        path: path.display().to_string(),
        batch,
        reason: format!(
            "a `{dtype}` value cannot be represented as `{}`",
            F::SCALAR_NAME
        ),
    })
}
