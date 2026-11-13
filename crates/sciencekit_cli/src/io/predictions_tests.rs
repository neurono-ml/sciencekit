//! Unit tests for the prediction and matrix writers.

use tempfile::tempdir;

use super::{SKOutputFormat, sk_write_predictions, sk_write_transformed_matrix};

#[test]
fn predictions_round_trip_through_csv_and_json() {
    let directory = tempdir().expect("a temporary directory");
    let csv = directory.path().join("predictions.csv");
    let json = directory.path().join("predictions.json");

    sk_write_predictions(&csv, SKOutputFormat::Csv, "y", &[1.5, 2.5]).expect("CSV writes");
    assert_eq!(
        std::fs::read_to_string(&csv).expect("CSV reads"),
        "y\n1.5\n2.5\n"
    );

    sk_write_predictions(&json, SKOutputFormat::Json, "y", &[1.5, 2.5]).expect("JSON writes");
    assert_eq!(
        std::fs::read_to_string(&json).expect("JSON reads"),
        "[[1.5],[2.5]]"
    );
}

#[test]
fn a_transformed_matrix_keeps_its_column_names() {
    let directory = tempdir().expect("a temporary directory");
    let path = directory.path().join("matrix.csv");
    let names = vec!["a".to_string(), "b".to_string()];
    let matrix = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    sk_write_transformed_matrix(&path, SKOutputFormat::Csv, &names, &matrix)
        .expect("the matrix writes");
    assert_eq!(
        std::fs::read_to_string(&path).expect("the matrix reads"),
        "a,b\n1,2\n3,4\n"
    );
}
