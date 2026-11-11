//! TDD suite for Arrow C Data and Stream Interface release-callback ownership
//! and `unsafe` reassembly guards (tasks 3.2-3.4).

#![allow(unsafe_code)]

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use arrow_array::ffi::{FFI_ArrowArray, FFI_ArrowSchema};
use arrow_array::ffi_stream::FFI_ArrowArrayStream;
use arrow_array::{ArrayRef, Float64Array, RecordBatch, RecordBatchIterator, RecordBatchReader};
use arrow_schema::{DataType, Field, Schema};

use super::{
    SKArrowFfiArray, SKArrowFfiStream, sk_export_array_to_ffi,
    sk_export_record_batch_reader_to_stream, sk_import_array_from_ffi, sk_import_stream,
};

type SKRawArrayRelease = unsafe extern "C" fn(*mut FFI_ArrowArray);
type SKRawStreamRelease = unsafe extern "C" fn(*mut FFI_ArrowArrayStream);

static ARRAY_RELEASES: AtomicUsize = AtomicUsize::new(0);
static SAVED_ARRAY_RELEASE: Mutex<Option<SKRawArrayRelease>> = Mutex::new(None);
static STREAM_RELEASES: AtomicUsize = AtomicUsize::new(0);
static SAVED_STREAM_RELEASE: Mutex<Option<SKRawStreamRelease>> = Mutex::new(None);
static RELEASE_LOCK: Mutex<()> = Mutex::new(());

unsafe extern "C" fn sk_counting_array_release(array: *mut FFI_ArrowArray) {
    ARRAY_RELEASES.fetch_add(1, Ordering::SeqCst);
    if array.is_null() {
        return;
    }
    let original = SAVED_ARRAY_RELEASE
        .lock()
        .expect("an uncontended saved release lock")
        .take();
    // SAFETY: `array` is a live C Data Interface pointer being released once;
    // chaining to Arrow's original callback frees the private allocation and
    // marks `release = None`.
    unsafe {
        match original {
            Some(callback) => {
                (*array).release = Some(callback);
                callback(array);
            }
            None => {
                (*array).release = None;
            }
        }
    }
}

unsafe extern "C" fn sk_counting_stream_release(stream: *mut FFI_ArrowArrayStream) {
    STREAM_RELEASES.fetch_add(1, Ordering::SeqCst);
    if stream.is_null() {
        return;
    }
    let original = SAVED_STREAM_RELEASE
        .lock()
        .expect("an uncontended saved stream release lock")
        .take();
    // SAFETY: `stream` is a live C Stream Interface pointer being released
    // once; chaining to Arrow's original callback drops the boxed reader and
    // marks `release = None`.
    unsafe {
        match original {
            Some(callback) => {
                (*stream).release = Some(callback);
                callback(stream);
            }
            None => {
                (*stream).release = None;
            }
        }
    }
}

fn sk_install_array_release_counter(mut array: FFI_ArrowArray) -> FFI_ArrowArray {
    *SAVED_ARRAY_RELEASE
        .lock()
        .expect("an uncontended saved release lock") = array.release;
    array.release = Some(sk_counting_array_release);
    array
}

fn sk_install_stream_release_counter(mut stream: FFI_ArrowArrayStream) -> FFI_ArrowArrayStream {
    *SAVED_STREAM_RELEASE
        .lock()
        .expect("an uncontended saved stream release lock") = stream.release;
    stream.release = Some(sk_counting_stream_release);
    stream
}

#[test]
fn sk_release_callback_runs_exactly_once_and_chains_original_destructor() {
    let _guard = RELEASE_LOCK.lock().expect("an uncontended lock");
    ARRAY_RELEASES.store(0, Ordering::SeqCst);

    let exported =
        sk_export_array_to_ffi(&Float64Array::from(vec![1.0, 2.0])).expect("an exported pair");
    let (array, schema) = exported.into_parts();
    let array = sk_install_array_release_counter(array);
    // SAFETY: `array` and `schema` were exported together by Arrow and the
    // custom release callback delegates to Arrow's original destructor.
    let pair = unsafe { SKArrowFfiArray::from_parts(array, schema) };
    let imported = sk_import_array_from_ffi(pair).expect("an import");
    assert_eq!(ARRAY_RELEASES.load(Ordering::SeqCst), 0);
    drop(imported);
    assert_eq!(ARRAY_RELEASES.load(Ordering::SeqCst), 1);

    let exported =
        sk_export_array_to_ffi(&Float64Array::from(vec![5.0])).expect("an exported pair");
    let (array, schema) = exported.into_parts();
    let array = sk_install_array_release_counter(array);
    // SAFETY: `array` and `schema` were exported together by Arrow.
    let pair = unsafe { SKArrowFfiArray::from_parts(array, schema) };
    drop(pair);
    assert_eq!(ARRAY_RELEASES.load(Ordering::SeqCst), 2);
}

#[test]
fn sk_failed_import_releases_the_exported_array_once() {
    let _guard = RELEASE_LOCK.lock().expect("an uncontended lock");
    ARRAY_RELEASES.store(0, Ordering::SeqCst);

    let array = sk_install_array_release_counter(FFI_ArrowArray::empty());
    let schema = FFI_ArrowSchema::try_from(&DataType::Float64).expect("a float schema");
    // SAFETY: the empty C array carries a valid no-op release callback and no
    // buffers; `sk_import_array_from_ffi` rejects zero-buffer arrays before
    // dereferencing any data pointer.
    let pair = unsafe { SKArrowFfiArray::from_parts(array, schema) };

    let error = sk_import_array_from_ffi(pair).expect_err("the handle is empty");
    assert!(error.to_string().contains("C Data Interface"), "{error}");
    assert!(error.to_string().contains("already released"), "{error}");
    assert_eq!(ARRAY_RELEASES.load(Ordering::SeqCst), 1);
}

#[test]
fn sk_stream_release_callback_runs_exactly_once_on_round_trip() {
    let _guard = RELEASE_LOCK.lock().expect("an uncontended lock");
    STREAM_RELEASES.store(0, Ordering::SeqCst);

    let schema = Arc::new(Schema::new(vec![Field::new(
        "value",
        DataType::Float64,
        false,
    )]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(Float64Array::from(vec![2.0, 4.0])) as ArrayRef],
    )
    .expect("a batch");
    let reader: Box<dyn RecordBatchReader + Send> = Box::new(RecordBatchIterator::new(
        vec![Ok(batch)].into_iter(),
        schema,
    ));

    let exported = sk_export_record_batch_reader_to_stream(reader).expect("an exported stream");
    let raw_stream = sk_install_stream_release_counter(exported.into_stream());
    // SAFETY: `raw_stream` was produced by `FFI_ArrowArrayStream::new` and the
    // custom release callback delegates to Arrow's original destructor.
    let stream = unsafe { SKArrowFfiStream::from_stream(raw_stream) };
    let imported_reader = sk_import_stream(stream).expect("an imported stream");

    let batches: Vec<RecordBatch> = imported_reader
        .map(|entry| entry.expect("a batch"))
        .collect();
    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 2);
    assert_eq!(STREAM_RELEASES.load(Ordering::SeqCst), 1);
}
