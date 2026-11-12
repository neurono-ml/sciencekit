//! Histogram-subtraction equivalence tests (task 2.3).

use ndarray::Array2;
use sciencekit_common::SKDataView;

use crate::histogram_boosting::binning::SKBinnedDataset;

use super::accumulate::sk_accumulate_histogram;
use super::subtract::sk_subtract_histogram;

#[test]
fn hist_subtract_equals_direct_accumulation() {
    let dense =
        Array2::from_shape_vec((6, 1), vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]).expect("fixture");
    let dataset =
        SKBinnedDataset::build(&SKDataView::Dense(dense.view()), &[], 8).expect("build");
    let gradients = [0.3, -1.2, 2.5, 0.7, -0.4, 1.1];
    let hessians = [0.9, 0.8, 1.0, 0.6, 0.7, 1.2];
    let all_rows: Vec<usize> = (0..6).collect();
    let smaller_rows = vec![0usize, 2, 4];
    let larger_rows = vec![1usize, 3, 5];
    let parent = sk_accumulate_histogram::<f64>(&dataset, &all_rows, &gradients, &hessians);
    let smaller = sk_accumulate_histogram::<f64>(&dataset, &smaller_rows, &gradients, &hessians);
    let direct_larger = sk_accumulate_histogram::<f64>(&dataset, &larger_rows, &gradients, &hessians);
    let subtracted_larger = sk_subtract_histogram(&parent, &smaller);
    let (left, right) = (
        subtracted_larger.feature(0),
        direct_larger.feature(0),
    );
    assert_eq!(left.total_count(), right.total_count());
    assert!((left.total_gradient() - right.total_gradient()).abs() < 1e-12);
    assert!((left.total_hessian() - right.total_hessian()).abs() < 1e-12);
    (0..8).for_each(|bin| {
        let bin = bin as u8;
        assert_eq!(left.count(bin), right.count(bin));
        assert!((left.gradient_sum(bin) - right.gradient_sum(bin)).abs() < 1e-12);
        assert!((left.hessian_sum(bin) - right.hessian_sum(bin)).abs() < 1e-12);
    });
}
