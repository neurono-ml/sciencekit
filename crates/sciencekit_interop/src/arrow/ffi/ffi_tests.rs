//! TDD suite for Arrow C Data and Stream Interface round trips and lifetime
//! independence (tasks 3.1-3.4).

use std::sync::Arc;

use arrow_array::types::Int32Type;
use arrow_array::{
    Array, ArrayRef, DictionaryArray, Float64Array, RecordBatch, RecordBatchIterator,
    RecordBatchReader, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use ndarray::array;

use super::{
    sk_export_array_to_ffi, sk_export_record_batch_reader_to_stream, sk_import_array_from_ffi,
    sk_import_stream,
};
use crate::arrow::sk_export_dense_matrix;

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

#[test]
fn sk_imported_array_survives_producer_drop() {
    let imported = {
        let producer = Float64Array::from(vec![10.5, 20.5, 30.5, 40.5]);
        let exported_pair = sk_export_array_to_ffi(&producer).expect("an exported C pair");
        let imported = sk_import_array_from_ffi(exported_pair).expect("an imported array");
        drop(producer);
        imported
    };

    let values = imported
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("a float64 array");
    assert_eq!(values.values().as_ref(), &[10.5, 20.5, 30.5, 40.5]);
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
