//! Export and import of single Arrow arrays through the C Data Interface
//! (spec `arrow-columnar-interop`).

use arrow_array::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use arrow_array::{Array, ArrayRef, make_array};
use sciencekit_common::SKError;

fn sk_ffi_error(context: &str, error: impl std::fmt::Display) -> SKError {
    SKError::Conversion(format!("Arrow C Data Interface {context} failed: {error}"))
}

/// An exported array paired with the schema that describes it.
///
/// The pair owns the C structures: dropping it calls each release callback
/// exactly once. Passing it to [`sk_import_array_from_ffi`] transfers that
/// ownership to the imported array instead.
pub struct SKArrowFfiArray {
    array: Option<FFI_ArrowArray>,
    schema: Option<FFI_ArrowSchema>,
}

impl SKArrowFfiArray {
    /// The exported C array, when it has not been released or imported yet.
    pub fn exported_array(&self) -> Option<&FFI_ArrowArray> {
        self.array.as_ref()
    }

    /// The exported C schema.
    pub fn schema(&self) -> &FFI_ArrowSchema {
        self.schema
            .as_ref()
            .expect("an exported pair always carries its schema")
    }

    /// Split the pair into its C structures, transferring ownership.
    pub fn into_parts(mut self) -> (FFI_ArrowArray, FFI_ArrowSchema) {
        let array = self
            .array
            .take()
            .expect("an exported pair always carries its array");
        let schema = self
            .schema
            .take()
            .expect("an exported pair always carries its schema");
        (array, schema)
    }

    /// Reassemble a pair from raw Arrow C Data Interface structures.
    ///
    /// # Safety
    /// The caller must guarantee that `array` and `schema` form a valid,
    /// matching Arrow C Data Interface pair whose buffers, lengths, offsets,
    /// children, dictionary pointers and release callbacks conform to the
    /// Arrow C Data Interface specification and have not already been
    /// consumed.
    pub unsafe fn from_parts(array: FFI_ArrowArray, schema: FFI_ArrowSchema) -> Self {
        SKArrowFfiArray {
            array: Some(array),
            schema: Some(schema),
        }
    }
}

/// Export one Arrow array as a C Data Interface pair.
///
/// # Errors
/// Returns [`SKError::Conversion`] when Arrow cannot convert the array's data
/// type to a C schema.
pub fn sk_export_array_to_ffi(array: &dyn Array) -> Result<SKArrowFfiArray, SKError> {
    let data = array.to_data();
    let (ffi_array, ffi_schema) =
        arrow_array::ffi::to_ffi(&data).map_err(|error| sk_ffi_error("export", error))?;
    Ok(SKArrowFfiArray {
        array: Some(ffi_array),
        schema: Some(ffi_schema),
    })
}

/// Import an array from a C Data Interface pair, taking ownership of its
/// release callback.
///
/// The imported array keeps the source buffers alive: releasing the producer
/// handle is the ownership transfer this call performs, so the data stays valid
/// for as long as the imported array lives.
///
/// # Errors
/// Returns [`SKError::Conversion`] when the pair is malformed or the schema and
/// array disagree.
pub fn sk_import_array_from_ffi(ffi: SKArrowFfiArray) -> Result<ArrayRef, SKError> {
    let (array, schema) = ffi.into_parts();
    let data_type =
        arrow_schema::DataType::try_from(&schema).map_err(|error| sk_ffi_error("import", error))?;
    // Reject a released or empty handle before Arrow reads its buffers: a
    // non-null array always carries at least one data buffer, and a null
    // release callback means the producer already gave the buffers up.
    if array.release.is_none()
        || (!matches!(data_type, arrow_schema::DataType::Null) && array.n_buffers <= 0)
    {
        return Err(sk_ffi_error(
            "import",
            "the C array is empty or was already released",
        ));
    }
    // SAFETY: a `SKArrowFfiArray` is constructed only by `sk_export_array_to_ffi`
    // or by `unsafe fn SKArrowFfiArray::from_parts` (whose caller guarantees a
    // valid, matching Arrow C Data Interface pair), and `into_parts` consumes
    // it once so `from_ffi` takes exclusive ownership of the buffers.
    let data = unsafe { arrow_array::ffi::from_ffi(array, &schema) }
        .map_err(|error| sk_ffi_error("import", error))?;
    Ok(make_array(data))
}
