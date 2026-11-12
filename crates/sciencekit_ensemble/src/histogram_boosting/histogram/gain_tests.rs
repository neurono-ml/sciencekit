//! Hand-computed numeric split gain tests (task 2.2).

use ndarray::Array2;
use sciencekit_common::SKDataView;

use crate::histogram_boosting::binning::SKBinnedDataset;

use super::accumulate::sk_accumulate_histogram;
use super::gain::{SKHistogramSplit, sk_scan_numeric_feature, sk_split_gain};

fn fixture(gradients: &[f64], _hessians: &[f64]) -> (SKBinnedDataset<f64>, Vec<usize>) {
    let rows = gradients.len();
    let values: Vec<f64> = (0..rows).map(|row| row as f64).collect();
    let dense = Array2::from_shape_vec((rows, 1), values).expect("fixture");
    let dataset =
        SKBinnedDataset::build(&SKDataView::Dense(dense.view()), &[], rows.max(2)).expect("build");
    (dataset, (0..rows).collect())
}

#[test]
fn hist_gain_matches_hand_computed_split() {
    let gradients = [1.0, 1.0, -1.0, -1.0];
    let hessians = [1.0; 4];
    let (dataset, rows) = fixture(&gradients, &hessians);
    let histogram = sk_accumulate_histogram(&dataset, &rows, &gradients, &hessians);
    let candidate = sk_scan_numeric_feature(0, histogram.feature(0), 1, 0.0).expect("split");
        assert!((candidate.gain - 4.0).abs() < 1e-12);
    match candidate.split {
        SKHistogramSplit::Numeric {
            maximum_bin,
            default_left,
            ..
        } => {
            assert_eq!(maximum_bin, 1);
            assert!(!default_left);
        }
        _ => panic!("expected a numeric split"),
    }
}

#[test]
fn hist_gain_applies_l2_regularization() {
    assert!((sk_split_gain::<f64>(2.0, 2.0, 0.0, 2.0, 2.0, 4.0, 0.0) - 1.0).abs() < 1e-12);
    assert!((sk_split_gain::<f64>(2.0, 2.0, 0.0, 2.0, 2.0, 4.0, 2.0) - 1.0 / 3.0).abs() < 1e-12);
}

#[test]
fn hist_gain_rejects_splits_below_minimum_samples_leaf() {
    let gradients = [1.0, 1.0, -1.0, -1.0];
    let hessians = [1.0; 4];
    let (dataset, rows) = fixture(&gradients, &hessians);
    let histogram = sk_accumulate_histogram(&dataset, &rows, &gradients, &hessians);
    assert!(sk_scan_numeric_feature(0, histogram.feature(0), 3, 0.0).is_none());
    assert!(sk_scan_numeric_feature(0, histogram.feature(0), 2, 0.0).is_some());
}

#[test]
fn hist_gain_evaluates_the_missing_direction() {
    let dense = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, f64::NAN, 3.0]).expect("fixture");
    let dataset =
        SKBinnedDataset::build(&SKDataView::Dense(dense.view()), &[], 4).expect("build");
    let gradients = [5.0, 5.0, 5.0, 0.1];
    let hessians = [1.0; 4];
    let rows: Vec<usize> = (0..4).collect();
    let histogram = sk_accumulate_histogram(&dataset, &rows, &gradients, &hessians);
    let candidate = sk_scan_numeric_feature(0, histogram.feature(0), 1, 0.0).expect("split");
    match candidate.split {
        SKHistogramSplit::Numeric { default_left, .. } => assert!(default_left),
        _ => panic!("expected a numeric split"),
    }
}
