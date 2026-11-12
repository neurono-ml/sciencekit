//! Export and import of Arrow record-batch streams through the C Stream
//! Interface (spec `arrow-columnar-interop`).

use arrow_array::RecordBatchReader;
use arrow_array::ffi_stream::{ArrowArrayStreamReader, FFI_ArrowArrayStream};
use sciencekit_common::SKError;

fn sk_stream_error(context: &str, error: impl std::fmt::Display) -> SKError {
    SKError::Conversion(format!(
        "Arrow C Stream Interface {context} failed: {error}"
    ))
}

/// An exported record-batch stream.
///
/// Dropping it calls the producer's release callback exactly once; importing
/// it with [`sk_import_stream`] transfers that ownership to the reader.
pub struct SKArrowFfiStream {
    stream: Option<FFI_ArrowArrayStream>,
}

impl SKArrowFfiStream {
    /// The exported C stream, when it has not been imported or released.
    pub fn exported_stream(&self) -> Option<&FFI_ArrowArrayStream> {
        self.stream.as_ref()
    }

    /// Split the wrapper into the C stream, transferring ownership.
    pub fn into_stream(mut self) -> FFI_ArrowArrayStream {
        self.stream
            .take()
            .expect("an exported stream always carries its handle")
    }

    /// Reassemble a wrapper from a raw Arrow C Stream Interface structure.
    ///
    /// # Safety
    /// The caller must guarantee that `stream` conforms to the Arrow C Stream
    /// Interface (`get_schema`, `get_next`, `get_last_error`, `release` and
    /// `private_data` form a valid, unconsumed producer stream).
    pub unsafe fn from_stream(stream: FFI_ArrowArrayStream) -> Self {
        SKArrowFfiStream {
            stream: Some(stream),
        }
    }
}

/// Export a record-batch reader as a C stream.
///
/// The reader's schema and batches are served through the C callbacks; the
/// producer-owned state is dropped when the consumer releases the stream.
///
/// # Errors
/// Returns [`SKError::Conversion`] only when the reader's schema cannot be
/// represented; construction itself is infallible.
pub fn sk_export_record_batch_reader_to_stream(
    reader: Box<dyn RecordBatchReader + Send>,
) -> Result<SKArrowFfiStream, SKError> {
    Ok(SKArrowFfiStream {
        stream: Some(FFI_ArrowArrayStream::new(reader)),
    })
}

/// Import a C stream as a record-batch reader, taking ownership of the stream.
///
/// # Errors
/// Returns [`SKError::Conversion`] when the stream is malformed.
pub fn sk_import_stream(ffi: SKArrowFfiStream) -> Result<ArrowArrayStreamReader, SKError> {
    ArrowArrayStreamReader::try_new(ffi.into_stream())
        .map_err(|error| sk_stream_error("import", error))
}
