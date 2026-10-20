//! Tests for `SKOneHotEncoder` (spec `one-hot-encoder`).

use ndarray::{Array2, array};
use sciencekit_common::SKBuilder;

use super::SKUnknownPolicy;
use super::core_implementation::SKOneHotEncoder;
use super::output::SKEncodedOutput;

fn text(rows: &[&[&str]]) -> Vec<Vec<String>> {
    rows.iter()
        .map(|row| row.iter().map(|label| (*label).to_owned()).collect())
        .collect()
}

/// Rebuild a borrowed nominal matrix from owned rows for the seam.
fn as_nominal(rows: &[Vec<String>]) -> Vec<Vec<&str>> {
    rows.iter()
        .map(|row| row.iter().map(String::as_str).collect())
        .collect()
}

/// Hand-checkable indicator matrix for a single text column.
#[test]
fn text_column_encodes_to_indicators() {
    let owned = text(&[&["a"], &["b"], &["a"]]);
    let view = as_nominal(&owned);
    let rows: Vec<&[&str]> = view.iter().map(Vec::as_slice).collect();

    let model = SKOneHotEncoder::build()
        .build()
        .unwrap()
        .fit(rows.as_slice())
        .unwrap();
    assert_eq!(model.number_of_output_features(), 2);
    let encoded: Array2<f64> = model.transform_dense(rows.as_slice()).unwrap();
    assert_eq!(encoded, array![[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]);
}

/// Output width is the sum of the per-column category counts.
#[test]
fn output_width_is_sum_of_category_counts() {
    let owned = text(&[&["a", "x"], &["b", "x"], &["a", "y"]]);
    let view = as_nominal(&owned);
    let rows: Vec<&[&str]> = view.iter().map(Vec::as_slice).collect();

    let model = SKOneHotEncoder::build()
        .build()
        .unwrap()
        .fit(rows.as_slice())
        .unwrap();
    // Column 0 has {a, b} = 2, column 1 has {x, y} = 2.
    assert_eq!(model.number_of_output_features(), 4);
    assert_eq!(model.number_of_input_columns(), 2);
}

/// Categories are assigned in first-occurrence order and fits are deterministic.
#[test]
fn first_occurrence_order_and_determinism() {
    let owned = text(&[&["b"], &["a"], &["b"], &["c"]]);
    let view = as_nominal(&owned);
    let rows: Vec<&[&str]> = view.iter().map(Vec::as_slice).collect();

    let first = SKOneHotEncoder::build()
        .build()
        .unwrap()
        .fit(rows.as_slice())
        .unwrap();
    let second = SKOneHotEncoder::build()
        .build()
        .unwrap()
        .fit(rows.as_slice())
        .unwrap();
    let first_table = first
        .column_categories(0)
        .unwrap()
        .as_label_table()
        .unwrap();
    let second_table = second
        .column_categories(0)
        .unwrap()
        .as_label_table()
        .unwrap();
    assert_eq!(first_table.index_of("b"), Some(0));
    assert_eq!(first_table.index_of("a"), Some(1));
    assert_eq!(first_table.index_of("c"), Some(2));
    assert_eq!(first_table, second_table);
}

/// A nominal table roundtrips label -> index -> label.
#[test]
fn table_roundtrip_nominal() {
    let owned = text(&[&["red"], &["green"]]);
    let view = as_nominal(&owned);
    let rows: Vec<&[&str]> = view.iter().map(Vec::as_slice).collect();

    let model = SKOneHotEncoder::build()
        .build()
        .unwrap()
        .fit(rows.as_slice())
        .unwrap();
    let table = model
        .column_categories(0)
        .unwrap()
        .as_label_table()
        .unwrap();
    let index = table.index_of("green").unwrap();
    assert_eq!(table.label_of(index), Some("green"));
}

/// Unknown categories are rejected by default.
#[test]
fn unknown_category_rejected_by_default() {
    let owned = text(&[&["a"], &["b"]]);
    let view = as_nominal(&owned);
    let rows: Vec<&[&str]> = view.iter().map(Vec::as_slice).collect();
    let model = SKOneHotEncoder::build()
        .build()
        .unwrap()
        .fit(rows.as_slice())
        .unwrap();

    let unknown = text(&[&["z"]]);
    let unknown_view = as_nominal(&unknown);
    let unknown_rows: Vec<&[&str]> = unknown_view.iter().map(Vec::as_slice).collect();
    let result: Result<Array2<f64>, _> = model.transform_dense(unknown_rows.as_slice());
    assert!(result.is_err());
}

/// Ignore encodes unknown categories as an all-zero block.
#[test]
fn unknown_category_ignored_encodes_zeros() {
    let owned = text(&[&["a"], &["b"]]);
    let view = as_nominal(&owned);
    let rows: Vec<&[&str]> = view.iter().map(Vec::as_slice).collect();
    let model = SKOneHotEncoder::build()
        .handle_unknown(SKUnknownPolicy::Ignore)
        .build()
        .unwrap()
        .fit(rows.as_slice())
        .unwrap();

    let unknown = text(&[&["z"]]);
    let unknown_view = as_nominal(&unknown);
    let unknown_rows: Vec<&[&str]> = unknown_view.iter().map(Vec::as_slice).collect();
    let encoded: Array2<f64> = model.transform_dense(unknown_rows.as_slice()).unwrap();
    assert_eq!(encoded, array![[0.0, 0.0]]);
}

/// Integer-coded columns encode, in first-occurrence order, and skip the
/// missing sentinel.
#[test]
fn integer_codes_skip_missing_sentinel() {
    let codes = Array2::from_shape_vec((3, 1), vec![0_i64, -1, 2]).unwrap();
    let model = SKOneHotEncoder::build()
        .build()
        .unwrap()
        .fit(codes.view())
        .unwrap();
    assert_eq!(model.number_of_output_features(), 2);
    let encoded: Array2<f64> = model.transform_dense(codes.view()).unwrap();
    assert_eq!(encoded, array![[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]);

    let table = model.column_categories(0).unwrap();
    assert_eq!(table.number_of_categories(), 2);
    assert_eq!(table.index_of_code(0), Some(0));
    assert_eq!(table.index_of_code(2), Some(1));
    assert_eq!(table.code_of(1), Some(2));
}

/// Sparse transform equals the dense transform exactly.
#[test]
fn sparse_matches_dense() {
    let owned = text(&[&["a", "x"], &["b", "y"], &["a", "x"]]);
    let view = as_nominal(&owned);
    let rows: Vec<&[&str]> = view.iter().map(Vec::as_slice).collect();
    let model = SKOneHotEncoder::build()
        .build()
        .unwrap()
        .fit(rows.as_slice())
        .unwrap();

    let dense: Array2<f64> = model.transform_dense(rows.as_slice()).unwrap();
    let sparse = model.transform_sparse::<f64, _>(rows.as_slice()).unwrap();
    let densified = sparse.to_dense();
    assert_eq!(densified, dense);
}

/// The builder selects the default output representation.
#[test]
fn builder_selects_default_output() {
    let owned = text(&[&["a"], &["b"]]);
    let view = as_nominal(&owned);
    let rows: Vec<&[&str]> = view.iter().map(Vec::as_slice).collect();

    let sparse_model = SKOneHotEncoder::build()
        .build()
        .unwrap()
        .fit(rows.as_slice())
        .unwrap();
    match sparse_model.transform::<f64, _>(rows.as_slice()).unwrap() {
        SKEncodedOutput::Sparse(_) => {}
        other => panic!("expected sparse by default, got {other:?}"),
    }

    let dense_model = SKOneHotEncoder::build()
        .sparse_output(false)
        .build()
        .unwrap()
        .fit(rows.as_slice())
        .unwrap();
    match dense_model.transform::<f64, _>(rows.as_slice()).unwrap() {
        SKEncodedOutput::Dense(_) => {}
        other => panic!("expected dense, got {other:?}"),
    }
}

/// Empty input is rejected before processing any element.
#[test]
fn empty_input_errors() {
    let rows: Vec<&[&str]> = Vec::new();
    let result = SKOneHotEncoder::build()
        .build()
        .unwrap()
        .fit(rows.as_slice());
    assert!(result.is_err());
}

/// A transform with an unexpected column count fails.
#[test]
fn transform_column_mismatch_errors() {
    let owned = text(&[&["a"], &["b"]]);
    let view = as_nominal(&owned);
    let rows: Vec<&[&str]> = view.iter().map(Vec::as_slice).collect();
    let model = SKOneHotEncoder::build()
        .build()
        .unwrap()
        .fit(rows.as_slice())
        .unwrap();

    let wider = text(&[&["a", "b"]]);
    let wider_view = as_nominal(&wider);
    let wider_rows: Vec<&[&str]> = wider_view.iter().map(Vec::as_slice).collect();
    let result: Result<Array2<f64>, _> = model.transform_dense(wider_rows.as_slice());
    assert!(result.is_err());
}

/// The backend defaults to Faer and flows into the fitted model.
#[test]
fn backend_default_and_override() {
    use sciencekit_common::SKBackendKind;

    let owned = text(&[&["a"]]);
    let view = as_nominal(&owned);
    let rows: Vec<&[&str]> = view.iter().map(Vec::as_slice).collect();

    let default_model = SKOneHotEncoder::build()
        .build()
        .unwrap()
        .fit(rows.as_slice())
        .unwrap();
    assert_eq!(default_model.backend(), SKBackendKind::Faer);

    let custom = SKOneHotEncoder::build()
        .backend(SKBackendKind::MatrixMultiply)
        .build()
        .unwrap()
        .fit(rows.as_slice())
        .unwrap();
    assert_eq!(custom.backend(), SKBackendKind::MatrixMultiply);
}

/// Both supported output scalars work.
#[test]
fn both_supported_floats() {
    let owned = text(&[&["a"], &["b"]]);
    let view = as_nominal(&owned);
    let rows: Vec<&[&str]> = view.iter().map(Vec::as_slice).collect();
    let model = SKOneHotEncoder::build()
        .build()
        .unwrap()
        .fit(rows.as_slice())
        .unwrap();

    let as_f32: Array2<f32> = model.transform_dense(rows.as_slice()).unwrap();
    let as_f64: Array2<f64> = model.transform_dense(rows.as_slice()).unwrap();
    assert_eq!(as_f32[[0, 0]], 1.0_f32);
    assert_eq!(as_f64[[0, 0]], 1.0_f64);
}

/// Uneven streaming batches produce the same tables as the dense fit.
#[test]
fn streaming_uneven_batches_equal_dense() {
    let owned_first = text(&[&["a", "x"], &["b", "x"]]);
    let view_first = as_nominal(&owned_first);
    let rows_first: Vec<&[&str]> = view_first.iter().map(Vec::as_slice).collect();

    let owned_second = text(&[&["a", "y"]]);
    let view_second = as_nominal(&owned_second);
    let rows_second: Vec<&[&str]> = view_second.iter().map(Vec::as_slice).collect();

    let owned_all = text(&[&["a", "x"], &["b", "x"], &["a", "y"]]);
    let view_all = as_nominal(&owned_all);
    let rows_all: Vec<&[&str]> = view_all.iter().map(Vec::as_slice).collect();

    let dense = SKOneHotEncoder::build()
        .build()
        .unwrap()
        .fit(rows_all.as_slice())
        .unwrap();

    let batches: Vec<&[&[&str]]> = vec![rows_first.as_slice(), rows_second.as_slice()];
    let streamed = SKOneHotEncoder::build()
        .build()
        .unwrap()
        .fit_streaming(batches)
        .unwrap();

    assert_eq!(
        streamed.number_of_output_features(),
        dense.number_of_output_features()
    );
    let dense_out: Array2<f64> = dense.transform_dense(rows_all.as_slice()).unwrap();
    let streamed_out: Array2<f64> = streamed.transform_dense(rows_all.as_slice()).unwrap();
    assert_eq!(dense_out, streamed_out);
}

/// Inconsistent streaming column counts are rejected.
#[test]
fn streaming_inconsistent_columns_error() {
    let owned_first = text(&[&["a", "x"]]);
    let view_first = as_nominal(&owned_first);
    let rows_first: Vec<&[&str]> = view_first.iter().map(Vec::as_slice).collect();

    let owned_second = text(&[&["a"]]);
    let view_second = as_nominal(&owned_second);
    let rows_second: Vec<&[&str]> = view_second.iter().map(Vec::as_slice).collect();

    let batches: Vec<&[&[&str]]> = vec![rows_first.as_slice(), rows_second.as_slice()];
    let result = SKOneHotEncoder::build()
        .build()
        .unwrap()
        .fit_streaming(batches);
    assert!(result.is_err());
}

/// An empty streaming sequence is rejected.
#[test]
fn streaming_empty_sequence_error() {
    let batches: Vec<&[&[&str]]> = Vec::new();
    let result = SKOneHotEncoder::build()
        .build()
        .unwrap()
        .fit_streaming(batches);
    assert!(result.is_err());
}

/// Acceptance: large data encodes, and one model serves concurrent readers.
#[test]
fn large_data_and_concurrency() {
    // 50 000 rows, four columns, ten categories each, built by a closed form.
    let rows_count = 50_000;
    let labels: Vec<String> = (0..10).map(|index| format!("c{index}")).collect();
    let owned: Vec<Vec<String>> = (0..rows_count)
        .map(|row| {
            (0..4)
                .map(|column| labels[(row * 7 + column * 3) % 10].clone())
                .collect()
        })
        .collect();
    let view = as_nominal(&owned);
    let rows: Vec<&[&str]> = view.iter().map(Vec::as_slice).collect();

    let model = SKOneHotEncoder::build()
        .build()
        .unwrap()
        .fit(rows.as_slice())
        .unwrap();
    assert_eq!(model.number_of_output_features(), 40);

    let expected: Array2<f64> = model.transform_dense(rows.as_slice()).unwrap();
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..4)
            .map(|_| {
                let model = &model;
                let rows = rows.as_slice();
                scope.spawn(move || model.transform_dense::<f64, _>(rows).unwrap())
            })
            .collect();
        for handle in handles {
            let output = handle.join().unwrap();
            assert_eq!(output, expected);
        }
    });
}

/// The default execution intent is Automatic.
#[test]
fn default_execution_intent_is_automatic() {
    use sciencekit_common::SKExecutionMode;
    let encoder = SKOneHotEncoder::build().build().unwrap();
    assert_eq!(encoder.execution_intent(), SKExecutionMode::Automatic);
}

/// A transform whose representation differs from the fitted one is rejected.
#[test]
fn transform_representation_mismatch_errors() {
    use sciencekit_common::SKError;
    let owned = text(&[&["a"], &["b"]]);
    let view = as_nominal(&owned);
    let rows: Vec<&[&str]> = view.iter().map(Vec::as_slice).collect();
    let model = SKOneHotEncoder::build()
        .build()
        .unwrap()
        .fit(rows.as_slice())
        .unwrap();

    let codes = Array2::from_shape_vec((1, 1), vec![0_i64]).unwrap();
    let result: Result<Array2<f64>, _> = model.transform_dense(codes.view());
    assert!(matches!(
        result,
        Err(SKError::UnsupportedRepresentation { .. })
    ));
}

/// A ragged nominal row is rejected instead of panicking.
#[test]
fn ragged_nominal_transform_errors() {
    let owned = text(&[&["a"], &["b"]]);
    let view = as_nominal(&owned);
    let rows: Vec<&[&str]> = view.iter().map(Vec::as_slice).collect();
    let model = SKOneHotEncoder::build()
        .build()
        .unwrap()
        .fit(rows.as_slice())
        .unwrap();

    let first: &[&str] = &["a"];
    let second: &[&str] = &[];
    let ragged: &[&[&str]] = &[first, second];
    let result: Result<Array2<f64>, _> = model.transform_dense(ragged);
    assert!(result.is_err());
}
