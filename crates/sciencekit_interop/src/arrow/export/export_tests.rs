//! TDD suite for exporting predictions, blocks and sparse matrices (tasks
//! 5.1-5.3).

use arrow_array::{Array, Float64Array};
use arrow_schema::DataType;
use ndarray::{Array1, array};
use sciencekit_common::{SKDataBatch, SKSupervisedBatch};

use super::{
    SK_ARROW_NUMBER_OF_COLUMNS_METADATA_KEY, SK_ARROW_NUMBER_OF_ROWS_METADATA_KEY,
    SKArrowBatchExport, sk_export_csr_matrix, sk_export_data_batch, sk_export_dense_matrix,
    sk_export_prediction, sk_export_supervised_batch,
};
use crate::arrow::sk_arrow_conversion_report;

fn csr_matrix() -> sprs::CsMat<f64> {
    let mut triplets = sprs::TriMat::new((2, 3));
    triplets.add_triplet(0, 0, 1.0);
    triplets.add_triplet(0, 2, 2.0);
    triplets.add_triplet(1, 1, 3.0);
    triplets.to_csr()
}

#[test]
fn sk_prediction_vector_exports_as_a_floating_arrow_column() {
    let prediction = Array1::from(vec![0.5, 1.5, 2.5]);
    let exported = sk_export_prediction(prediction.view()).expect("an exported prediction");

    let array = exported
        .array()
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("a float64 array");
    assert_eq!(array.values().as_ref(), &[0.5, 1.5, 2.5]);
    assert_eq!(array.len(), 3);
    assert!(
        !exported.is_zero_copy(),
        "export copies into an Arrow buffer"
    );
}

#[test]
fn sk_dense_data_batch_exports_feature_columns() {
    let batch = SKDataBatch::new_dense(array![[1.0, 2.0], [3.0, 4.0]], 0, true);
    let exported = sk_export_data_batch(&batch).expect("an exported block");

    let SKArrowBatchExport::Dense(exported) = exported else {
        panic!("a dense block must export as a record batch");
    };
    assert_eq!(exported.batch().num_rows(), 2);
    assert_eq!(exported.batch().num_columns(), 2);
    assert_eq!(exported.batch().schema().field(0).name(), "feature_0");
    assert_eq!(exported.batch().schema().field(1).name(), "feature_1");
    let first = exported
        .batch()
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("a float column");
    assert_eq!(first.values().as_ref(), &[1.0, 3.0]);
    assert!(!exported.is_zero_copy());
}

#[test]
fn sk_supervised_batch_exports_features_and_the_target_column() {
    let batch = SKSupervisedBatch::new_dense(
        array![[1.0], [3.0]],
        Array1::from(vec![10.0, 20.0]),
        0,
        true,
    )
    .expect("a supervised batch");
    let exported =
        sk_export_supervised_batch(&batch, "target").expect("an exported supervised block");

    assert_eq!(exported.batch().num_columns(), 2);
    assert_eq!(exported.batch().schema().field(1).name(), "target");
    let targets = exported
        .batch()
        .column(1)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("a float column");
    assert_eq!(targets.values().as_ref(), &[10.0, 20.0]);
}

#[test]
fn sk_csr_export_preserves_pointers_indices_values_and_shape() {
    let matrix = csr_matrix();
    let exported = sk_export_csr_matrix(matrix.view()).expect("an exported CSR block");

    let indptr = exported
        .indptr()
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .expect("an integer column");
    assert_eq!(indptr.values().as_ref(), &[0, 2, 3]);
    let indices = exported
        .indices()
        .as_any()
        .downcast_ref::<arrow_array::Int64Array>()
        .expect("an integer column");
    assert_eq!(indices.values().as_ref(), &[0, 2, 1]);
    let values = exported
        .data()
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("a float column");
    assert_eq!(values.values().as_ref(), &[1.0, 2.0, 3.0]);
    assert_eq!(exported.shape(), (2, 3));
    assert_eq!(
        exported
            .schema()
            .metadata()
            .get(SK_ARROW_NUMBER_OF_ROWS_METADATA_KEY),
        Some(&"2".to_string())
    );
    assert_eq!(
        exported
            .schema()
            .metadata()
            .get(SK_ARROW_NUMBER_OF_COLUMNS_METADATA_KEY),
        Some(&"3".to_string())
    );
}

#[test]
fn sk_sparse_data_batch_exports_as_a_csr_block() {
    let batch = SKDataBatch::new_sparse(csr_matrix(), 0, true);
    let exported = sk_export_data_batch(&batch).expect("an exported sparse block");

    let SKArrowBatchExport::Sparse(exported) = exported else {
        panic!("a sparse block must export as CSR arrays");
    };
    assert_eq!(exported.shape(), (2, 3));
}

#[test]
fn sk_both_scalar_precisions_export_and_cast_reports_zero_copy_false() {
    let single = sk_export_prediction(Array1::from(vec![1.5_f32]).view()).expect("a f32 export");
    assert_eq!(single.array().data_type(), &DataType::Float32);

    let double = sk_export_prediction(Array1::from(vec![1.5_f64]).view()).expect("a f64 export");
    assert_eq!(double.array().data_type(), &DataType::Float64);

    let cast = sk_arrow_conversion_report::<f64>(single.array().as_ref());
    assert!(!cast.is_zero_copy());
    assert!(cast.is_cast());

    assert!(!single.is_zero_copy());
    assert!(!double.is_zero_copy());
}

#[test]
fn sk_zero_column_and_column_major_dense_matrices_export_cleanly() {
    let zero_columns = ndarray::Array2::<f64>::zeros((3, 0));
    let empty_export =
        sk_export_dense_matrix(zero_columns.view(), "empty").expect("zero-column matrix exports");
    assert_eq!(empty_export.len(), 3);

    let transposed = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0]];
    let view = transposed.t();
    assert!(!view.is_standard_layout());
    let exported = sk_export_dense_matrix(view, "feature").expect("column-major view exports");
    let structure = exported
        .as_any()
        .downcast_ref::<arrow_array::StructArray>()
        .expect("a struct array");
    assert_eq!(structure.len(), 3);
    assert_eq!(structure.num_columns(), 2);
    let first = structure
        .column(0)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("a float64 column");
    assert_eq!(first.values().as_ref(), &[1.0, 2.0, 3.0]);
}

#[test]
fn sk_sparse_supervised_batch_export_returns_unsupported_representation() {
    let batch =
        SKSupervisedBatch::new_sparse(csr_matrix(), Array1::from(vec![10.0, 20.0]), 0, true)
            .expect("a sparse supervised batch");
    let error = sk_export_supervised_batch(&batch, "target")
        .expect_err("sparse supervised blocks have no single rectangular RecordBatch shape");
    match error {
        sciencekit_common::SKError::UnsupportedRepresentation {
            representation,
            suggestion,
        } => {
            assert_eq!(representation, "sparse_supervised");
            assert!(suggestion.contains("sk_export_data_batch"), "{suggestion}");
        }
        other => panic!("expected UnsupportedRepresentation, got {other:?}"),
    }
}
