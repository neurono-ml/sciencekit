//! Hand-computed gradient/Hessian histogram accumulation tests (task 2.1).

use ndarray::Array2;
use sciencekit_common::SKDataView;

use crate::histogram_boosting::binning::SKBinnedDataset;

use super::accumulate::sk_accumulate_histogram;

#[test]
fn hist_accumulate_sums_gradients_and_hessians_per_bin() {
    let dense = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).expect("fixture");
    let dataset =
        SKBinnedDataset::build(&SKDataView::Dense(dense.view()), &[], 4).expect("build");
    let gradients = [1.0, 2.0, 3.0, 4.0];
    let hessians = [0.5, 0.5, 0.5, 0.5];
    let rows: Vec<usize> = (0..4).collect();
    let histogram = sk_accumulate_histogram(&dataset, &rows, &gradients, &hessians);
    let feature = histogram.feature(0);
    let occupied: Vec<u8> = rows
        .iter()
        .map(|&row| dataset.bin(0, row))
        .collect();
    assert_eq!(occupied, vec![0, 1, 2, 3]);
    occupied.iter().enumerate().for_each(|(row, &bin)| {
        assert_eq!(feature.gradient_sum(bin), gradients[row]);
        assert_eq!(feature.hessian_sum(bin), hessians[row]);
        assert_eq!(feature.count(bin), 1);
    });
    assert_eq!(feature.total_gradient(), 10.0);
    assert_eq!(feature.total_hessian(), 2.0);
    assert_eq!(feature.total_count(), 4);
}

#[test]
fn hist_accumulate_respects_the_row_subset() {
    let dense = Array2::from_shape_vec((4, 1), vec![0.0, 1.0, 2.0, 3.0]).expect("fixture");
    let dataset =
        SKBinnedDataset::build(&SKDataView::Dense(dense.view()), &[], 4).expect("build");
    let gradients = [1.0, 2.0, 3.0, 4.0];
    let hessians = [1.0; 4];
    let histogram = sk_accumulate_histogram(&dataset, &[1, 3], &gradients, &hessians);
    assert_eq!(histogram.feature(0).total_count(), 2);
    assert_eq!(histogram.feature(0).total_gradient(), 6.0);
    assert_eq!(histogram.feature(0).count(0), 0);
    assert_eq!(histogram.feature(0).count(1), 1);
}
