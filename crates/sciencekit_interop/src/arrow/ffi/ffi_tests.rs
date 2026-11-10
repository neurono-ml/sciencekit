//! TDD suite for the Arrow C Data Interface (tasks 3.1-3.4).
//!
//! The release-counting tests install a custom callback, so `unsafe` is
//! allowed here and confined to that callback.

#![allow(unsafe_code)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use arrow_array::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use arrow_array::types::Int32Type;
use arrow_array::{
    Array, ArrayRef, DictionaryArray, Float64Array, RecordBatch, RecordBatchIterator,
    RecordBatchReader, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use ndarray::array;

use super::{
    SKArrowFfiArray, sk_export_array_to_ffi, sk_export_record_batch_reader_to_stream,
    sk_import_array_from_ffi, sk_import_stream,
};
use crate::arrow::sk_export_dense_matrix;

static RELEASES: AtomicUsize = AtomicUsize::new(0);
static RELEASE_LOCK: Mutex<()> = Mutex::new(());

unsafe extern "C" fn sk_counting_release(array: *mut FFI_ArrowArray) {
    RELEASES.fetch_add(1, Ordering::SeqCst);
    if array.is_null() {
        return;
    }
    // Disarm the callback so a second release cannot double free; the test
    // intentionally leaks the exported buffers.
    unsafe {
        (*array).release = None;
    }
}

fn sk_dense_matrix() -> ndarray::Array2<f64> {
    array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
}

#[test]
fn sk_dense_matrix_round_trips_through_the_c_data_interface() {
    let matrix = sk_dense_matrix();
    let exported = sk_export_dense_matrix(matrix.view(), "prediction").expect("an export");

    let ffi = sk_export_array_to_ffi(exported.as_ref()).expect("a C pair");
    let imported = sk_import_array_from_ffi(ffi).expect("an import");

    let structure = imported
        .as_any()
        .downcast_ref::<arrow_array::StructArray>()
        .expect("a struct array");
    assert_eq!(structure.len(), 3);
    assert_eq!(structure.num_columns(), 2);
    let first = structure
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("a float column");
    assert_eq!(first.values().as_ref(), &[1.0, 3.0, 5.0]);
    let second = structure
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("a float column");
    assert_eq!(second.values().as_ref(), &[2.0, 4.0, 6.0]);
}

fn sk_two_batch_reader() -> Box<dyn RecordBatchReader + Send> {
    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Float64,
        false,
    )]));
    let first = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Float64Array::from(vec![1.0, 2.0])) as ArrayRef],
    )
    .expect("a first batch");
    let second = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Float64Array::from(vec![3.0])) as ArrayRef],
    )
    .expect("a second batch");
    Box::new(RecordBatchIterator::new(
        vec![Ok(first), Ok(second)].into_iter(),
        schema,
    ))
}

#[test]
fn sk_stream_round_trips_batch_for_batch() {
    let stream =
        sk_export_record_batch_reader_to_stream(sk_two_batch_reader()).expect("an exported stream");
    let reader = sk_import_stream(stream).expect("an imported stream");

    let batches: Vec<RecordBatch> = reader.map(|batch| batch.expect("a batch")).collect();
    assert_eq!(batches.len(), 2);
    assert_eq!(batches[0].num_rows(), 2);
    assert_eq!(batches[1].num_rows(), 1);
    assert_eq!(batches[0].schema().field(0).name(), "value");
    let values = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("a float column");
    assert_eq!(values.values().as_ref(), &[1.0, 2.0]);
}

#[test]
fn sk_release_callback_runs_exactly_once() {
    let _guard = RELEASE_LOCK.lock().expect("an uncontended lock");
    RELEASES.store(0, Ordering::SeqCst);

    // Import takes ownership: the callback runs when the imported array drops.
    let exported =
        sk_export_array_to_ffi(&Float64Array::from(vec![1.0, 2.0])).expect("an exported pair");
    let (mut array, schema) = exported.into_parts();
    array.release = Some(sk_counting_release);
    let pair = SKArrowFfiArray::from_parts(array, schema);
    let imported = sk_import_array_from_ffi(pair).expect("an import");
    assert_eq!(RELEASES.load(Ordering::SeqCst), 0);
    drop(imported);
    assert_eq!(RELEASES.load(Ordering::SeqCst), 1);

    // Dropping an unimported pair releases once as well.
    let exported =
        sk_export_array_to_ffi(&Float64Array::from(vec![5.0])).expect("an exported pair");
    let (mut array, schema) = exported.into_parts();
    array.release = Some(sk_counting_release);
    let pair = SKArrowFfiArray::from_parts(array, schema);
    drop(pair);
    assert_eq!(RELEASES.load(Ordering::SeqCst), 2);
}

#[test]
fn sk_failed_import_releases_the_exported_array_once() {
    let _guard = RELEASE_LOCK.lock().expect("an uncontended lock");
    RELEASES.store(0, Ordering::SeqCst);

    // An empty C array declares zero buffers while the schema declares one, so
    // the import fails before touching any buffer, and the ownership transfer
    // still releases the producer handle exactly once.
    let mut array = FFI_ArrowArray::empty();
    array.release = Some(sk_counting_release);
    let schema = FFI_ArrowSchema::try_from(&DataType::Float64).expect("a float schema");
    let pair = SKArrowFfiArray::from_parts(array, schema);

    let error = sk_import_array_from_ffi(pair).expect_err("the handle is empty");
    assert!(error.to_string().contains("C Data Interface"), "{error}");
    assert!(error.to_string().contains("already released"), "{error}");
    assert_eq!(RELEASES.load(Ordering::SeqCst), 1);
}

#[test]
fn sk_dictionary_and_text_round_trip_through_the_c_data_interface() {
    let dictionary =
        DictionaryArray::<Int32Type>::from_iter(vec![Some("cat"), Some("dog"), Some("cat"), None]);
    let exported = sk_export_array_to_ffi(&dictionary).expect("a dictionary pair");
    let imported = sk_import_array_from_ffi(exported).expect("a dictionary import");
    assert!(matches!(imported.data_type(), DataType::Dictionary(_, _)));
    let imported_dictionary = imported
        .as_any()
        .downcast_ref::<DictionaryArray<Int32Type>>()
        .expect("a dictionary array");
    assert_eq!(imported_dictionary.len(), 4);
    assert!(imported_dictionary.is_null(3));

    let text = StringArray::from(vec![Some("alpha"), None, Some("beta")]);
    let exported = sk_export_array_to_ffi(&text).expect("a text pair");
    let imported = sk_import_array_from_ffi(exported).expect("a text import");
    let imported_text = imported
        .as_any()
        .downcast_ref::<StringArray>()
        .expect("a text array");
    assert_eq!(imported_text.value(0), "alpha");
    assert!(imported_text.is_null(1));
    assert_eq!(imported_text.value(2), "beta");
}
