//! Sequential blocks over a random-access source (spec `memory-mapped-sources`).
//!
//! Every mapped source can also feed the streaming driver: [`sk_next_mapped_block`]
//! copies at most `block_size` rows into an owned [`SKDataBatch`]. The block is
//! owned because streaming batches survive the source advance, while random
//! access keeps returning borrowed rows.

use std::sync::atomic::{AtomicUsize, Ordering};

use ndarray::Array2;
use sciencekit_common::{SKDataBatch, SKError, SKFloat, SKMappableSource};

/// Copy the next bounded block of rows into an owned streaming batch.
///
/// `cursor` holds the next row to copy; the function advances it and marks the
/// batch final when it reaches the last row.
///
/// # Errors
/// Propagates the source's row error and rejects a zero block size.
pub(crate) fn sk_next_mapped_block<F, Source>(
    source: &Source,
    cursor: &AtomicUsize,
    block_size: usize,
) -> Result<Option<SKDataBatch<F>>, SKError>
where
    F: SKFloat,
    Source: SKMappableSource<F, Error = SKError>,
{
    if block_size == 0 {
        return Err(SKError::InvalidHyperparameter {
            name: "block_size",
            reason: "a streaming block must contain at least one row".to_string(),
        });
    }
    let number_of_rows = source.number_of_rows();
    let start = cursor.load(Ordering::Relaxed);
    if start >= number_of_rows {
        return Ok(None);
    }
    let end = (start + block_size).min(number_of_rows);
    let number_of_columns = source.row(start)?.len();
    let mut data = Array2::<F>::zeros((end - start, number_of_columns));
    for (offset, index) in (start..end).enumerate() {
        data.row_mut(offset).assign(&source.row(index)?);
    }
    cursor.store(end, Ordering::Relaxed);
    Ok(Some(SKDataBatch::new_dense(
        data,
        start / block_size,
        end == number_of_rows,
    )))
}
