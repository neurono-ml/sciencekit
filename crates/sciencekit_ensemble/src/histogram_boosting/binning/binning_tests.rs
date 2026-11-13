//! Quantile edge and missing-bin tests (task 1.2).

use super::edges::SKNumericBinning;

#[test]
fn hist_binning_derives_at_most_maximum_bins() {
    let values: Vec<f64> = (0..1_000).map(|value| value as f64).collect();
    let binning = SKNumericBinning::derive(values.iter().copied(), 16);
    assert!(binning.number_of_bins() <= 16);
    assert!(binning.edges().len() < 16);
    assert_eq!(binning.bin(0.0), 0);
    assert_eq!(
        usize::from(binning.bin(999.0)),
        binning.number_of_bins() - 1
    );
    assert!(binning.bin(400.0) < binning.bin(600.0));
}

#[test]
fn hist_binning_routes_missing_to_dedicated_bin() {
    let values: Vec<f64> = vec![1.0, 2.0, f64::NAN, 4.0, 5.0];
    let binning = SKNumericBinning::derive(values.iter().copied(), 8);
    assert!(usize::from(binning.bin(f64::NAN)) > binning.number_of_bins() - 1);
    assert_eq!(binning.bin(f64::NAN), binning.missing_bin());
    assert_ne!(binning.bin(2.0), binning.missing_bin());
}

#[test]
fn hist_binning_is_deterministic() {
    let values: Vec<f64> = (0..10_000)
        .map(|value| ((value * 37) % 1013) as f64 / 7.0)
        .collect();
    let first = SKNumericBinning::derive(values.iter().copied(), 32);
    let second = SKNumericBinning::derive(values.iter().copied(), 32);
    assert_eq!(first.edges(), second.edges());
    assert_eq!(first.bin(f64::NAN), second.bin(f64::NAN));
    assert_eq!(first.number_of_bins(), second.number_of_bins());
}
