//! Arrow IPC footer indexing (spec `memory-mapped-sources`).
//!
//! Opening reads the footer and each record batch's message metadata only: the
//! body buffers stay untouched until a row is requested. The index stores the
//! first row and row count of every batch, so a lookup is a binary search plus
//! the decode of the one batch that contains the row.

use arrow_ipc::convert::fb_to_schema;
use arrow_schema::{DataType, Schema};
use sciencekit_common::SKError;

const ARROW_MAGIC: &[u8; 6] = b"ARROW1";
const CONTINUATION_MARKER: &[u8; 4] = &[0xff, 0xff, 0xff, 0xff];

/// One indexed record batch: where it starts and how many rows it holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct SKArrowIpcBatch {
    /// The row index of the batch's first row.
    pub(crate) first_row: usize,
    /// The number of rows in the batch.
    pub(crate) number_of_rows: usize,
}

/// The footer-derived layout of an Arrow IPC file.
#[derive(Debug, Clone)]
pub(crate) struct SKArrowIpcLayout {
    pub(crate) schema: Schema,
    pub(crate) batches: Vec<SKArrowIpcBatch>,
    pub(crate) number_of_rows: usize,
}

impl SKArrowIpcLayout {
    /// The number of indexed batches.
    pub(crate) fn number_of_batches(&self) -> usize {
        self.batches.len()
    }

    /// The index of the batch that contains `row`.
    pub(crate) fn batch_index_for(&self, row: usize) -> usize {
        self.batches
            .partition_point(|batch| batch.first_row <= row)
            .saturating_sub(1)
    }
}

/// Read an Arrow IPC footer and index its record batches without decoding data.
pub(crate) fn sk_read_ipc_layout(bytes: &[u8], path: &str) -> Result<SKArrowIpcLayout, SKError> {
    if bytes.len() < 16 || &bytes[..6] != ARROW_MAGIC || &bytes[bytes.len() - 6..] != ARROW_MAGIC {
        return Err(SKError::MalformedMappedHeader {
            path: path.to_string(),
            format: "arrow-ipc",
            reason: "the file does not start and end with the `ARROW1` magic".to_string(),
        });
    }
    let trailer: [u8; 10] = bytes[bytes.len() - 10..]
        .try_into()
        .expect("the trailer is ten bytes");
    let footer_length = arrow_ipc::reader::read_footer_length(trailer).map_err(|error| {
        SKError::MalformedMappedHeader {
            path: path.to_string(),
            format: "arrow-ipc",
            reason: error.to_string(),
        }
    })?;
    let footer_start = bytes.len().checked_sub(10 + footer_length).ok_or_else(|| {
        SKError::MalformedMappedHeader {
            path: path.to_string(),
            format: "arrow-ipc",
            reason: "the declared footer length exceeds the file".to_string(),
        }
    })?;
    let footer =
        arrow_ipc::root_as_footer(&bytes[footer_start..bytes.len() - 10]).map_err(|error| {
            SKError::MalformedMappedHeader {
                path: path.to_string(),
                format: "arrow-ipc",
                reason: error.to_string(),
            }
        })?;
    let schema =
        footer
            .schema()
            .map(fb_to_schema)
            .ok_or_else(|| SKError::MalformedMappedHeader {
                path: path.to_string(),
                format: "arrow-ipc",
                reason: "the footer declares no schema".to_string(),
            })?;
    for field in schema.fields() {
        if !matches!(field.data_type(), DataType::Float32 | DataType::Float64) {
            return Err(SKError::UnsupportedMappedDtype {
                path: path.to_string(),
                dtype: field.data_type().to_string(),
                supported: "Float32, Float64".to_string(),
            });
        }
    }
    let blocks = footer
        .recordBatches()
        .ok_or_else(|| SKError::MalformedMappedHeader {
            path: path.to_string(),
            format: "arrow-ipc",
            reason: "the footer declares no record batches".to_string(),
        })?;
    let mut batches = Vec::with_capacity(blocks.len());
    let mut number_of_rows = 0;
    for (index, block) in blocks.iter().enumerate() {
        let offset =
            usize::try_from(block.offset()).map_err(|_| SKError::MalformedMappedHeader {
                path: path.to_string(),
                format: "arrow-ipc",
                reason: format!("batch {index} has a negative offset"),
            })?;
        let metadata_length = block.metaDataLength() as usize;
        let body_length = block.bodyLength() as usize;
        offset
            .checked_add(metadata_length)
            .and_then(|end| end.checked_add(body_length))
            .filter(|end| *end <= bytes.len())
            .ok_or_else(|| SKError::MalformedMappedHeader {
                path: path.to_string(),
                format: "arrow-ipc",
                reason: format!("batch {index} extends past the end of the file"),
            })?;
        let metadata = sk_batch_metadata(&bytes[offset..offset + metadata_length], path, index)?;
        batches.push(SKArrowIpcBatch {
            first_row: number_of_rows,
            number_of_rows: metadata,
        });
        number_of_rows += metadata;
    }
    Ok(SKArrowIpcLayout {
        schema,
        batches,
        number_of_rows,
    })
}

/// Read one batch's row count from its message metadata, never its body.
fn sk_batch_metadata(block: &[u8], path: &str, index: usize) -> Result<usize, SKError> {
    let message_bytes = if block.len() >= 8 && &block[..4] == CONTINUATION_MARKER {
        &block[8..]
    } else if block.len() >= 4 {
        &block[4..]
    } else {
        return Err(SKError::UnreadableMappedBatch {
            path: path.to_string(),
            batch: index,
            reason: "the metadata block is shorter than the encapsulation prefix".to_string(),
        });
    };
    let message = arrow_ipc::root_as_message(message_bytes).map_err(|error| {
        SKError::UnreadableMappedBatch {
            path: path.to_string(),
            batch: index,
            reason: error.to_string(),
        }
    })?;
    let batch = message
        .header_as_record_batch()
        .ok_or_else(|| SKError::UnreadableMappedBatch {
            path: path.to_string(),
            batch: index,
            reason: "the message is not a record batch".to_string(),
        })?;
    usize::try_from(batch.length()).map_err(|_| SKError::UnreadableMappedBatch {
        path: path.to_string(),
        batch: index,
        reason: "the batch declares a negative row count".to_string(),
    })
}
