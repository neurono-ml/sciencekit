//! Unit tests for the dataset adapter.

use tempfile::tempdir;

use super::sk_load_dataset;

#[test]
fn csv_dataset_loads_features_and_target_in_column_order() {
    let directory = tempdir().expect("a temporary directory");
    let path = directory.path().join("data.csv");
    std::fs::write(&path, "a,b,y\n1.0,10.0,0.0\n2.0,20.0,1.0\n").expect("the CSV");
    let dataset = sk_load_dataset(&path, None, Some("y")).expect("the dataset loads");
    assert_eq!(dataset.number_of_rows(), 2);
    assert_eq!(dataset.number_of_features(), 2);
    assert_eq!(dataset.feature_names, vec!["a", "b"]);
    assert_eq!(dataset.targets.expect("a target")[1], 1.0);
}

#[test]
fn an_unknown_target_is_a_validation_error() {
    let directory = tempdir().expect("a temporary directory");
    let path = directory.path().join("data.csv");
    std::fs::write(&path, "a,b\n1.0,10.0\n").expect("the CSV");
    let error = sk_load_dataset(&path, None, Some("absent")).expect_err("the target is absent");
    assert_eq!(error.exit_code(), crate::cli::exit_codes::SKExitCode::Usage);
}

#[test]
fn the_format_is_inferred_from_the_extension_and_validated_by_name() {
    use super::SKDatasetFormat;
    use std::path::Path;

    assert_eq!(SKDatasetFormat::from_path(Path::new("data.parquet")), SKDatasetFormat::Parquet);
    assert_eq!(SKDatasetFormat::from_path(Path::new("data.csv")), SKDatasetFormat::Csv);
    assert_eq!(SKDatasetFormat::from_name("CSV").expect("csv"), SKDatasetFormat::Csv);
    assert!(SKDatasetFormat::from_name("arff").is_err());
}
