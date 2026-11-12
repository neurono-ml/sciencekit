//! Categorical partition tests (task 2.4).

use ndarray::Array2;
use sciencekit_common::SKDataView;

use crate::histogram_boosting::binning::SKBinnedDataset;

use super::accumulate::sk_accumulate_histogram;
use super::categorical::sk_scan_categorical_feature;
use super::gain::SKHistogramSplit;

#[test]
fn hist_categorical_partitions_categories_by_gradient_ratio() {
    let dense = Array2::from_shape_vec((3, 1), vec![10.0, 20.0, 30.0]).expect("fixture");
    let dataset =
        SKBinnedDataset::build(&SKDataView::Dense(dense.view()), &[0], 8).expect("build");
    let gradients = [3.0, -4.0, 3.0];
    let hessians = [1.0; 3];
    let rows: Vec<usize> = (0..3).collect();
    let histogram = sk_accumulate_histogram::<f64>(&dataset, &rows, &gradients, &hessians);
    let candidate =
        sk_scan_categorical_feature(0, histogram.feature(0), 3, 1, 0.0).expect("partition");
    assert!((candidate.gain - 98.0 / 3.0).abs() < 1e-12);
    match candidate.split {
        SKHistogramSplit::Categorical {
            left_bins,
            unknown_left,
            ..
        } => {
            assert!(left_bins[0]);
            assert!(!left_bins[1]);
            assert!(left_bins[2]);
            assert!(!unknown_left);
        }
        _ => panic!("expected a categorical split"),
    }
}

#[test]
fn hist_categorical_respects_minimum_samples_leaf() {
    let dense = Array2::from_shape_vec((3, 1), vec![10.0, 20.0, 30.0]).expect("fixture");
    let dataset =
        SKBinnedDataset::build(&SKDataView::Dense(dense.view()), &[0], 8).expect("build");
    let gradients = [3.0, -4.0, 3.0];
    let hessians = [1.0; 3];
    let rows: Vec<usize> = (0..3).collect();
    let histogram = sk_accumulate_histogram::<f64>(&dataset, &rows, &gradients, &hessians);
    assert!(sk_scan_categorical_feature(0, histogram.feature(0), 3, 2, 0.0).is_none());
}
