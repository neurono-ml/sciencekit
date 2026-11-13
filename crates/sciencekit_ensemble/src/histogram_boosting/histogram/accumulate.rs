//! Per-feature gradient/Hessian/count accumulation, parallel across features
//! (spec `histogram-gradient-boosting`, design decision 3).

use rayon::prelude::*;
use sciencekit_common::SKFloat;

use crate::histogram_boosting::binning::SKBinnedDataset;

/// Gradient, Hessian and sample-count sums of one feature's histogram.
#[derive(Debug, Clone, PartialEq)]
pub struct SKFeatureHistogram<F> {
    pub(crate) gradient_sums: Vec<F>,
    pub(crate) hessian_sums: Vec<F>,
    pub(crate) counts: Vec<usize>,
    pub(crate) total_gradient: F,
    pub(crate) total_hessian: F,
    pub(crate) total_count: usize,
}

impl<F: SKFloat> SKFeatureHistogram<F> {
    fn empty(stride: usize) -> Self {
        Self {
            gradient_sums: vec![F::zero(); stride],
            hessian_sums: vec![F::zero(); stride],
            counts: vec![0; stride],
            total_gradient: F::zero(),
            total_hessian: F::zero(),
            total_count: 0,
        }
    }

    /// Sum of negative gradients in `bin`.
    pub fn gradient_sum(&self, bin: u8) -> F {
        self.gradient_sums[usize::from(bin)]
    }

    /// Sum of Hessians in `bin`.
    pub fn hessian_sum(&self, bin: u8) -> F {
        self.hessian_sums[usize::from(bin)]
    }

    /// Number of node rows in `bin`.
    pub fn count(&self, bin: u8) -> usize {
        self.counts[usize::from(bin)]
    }

    /// Sum of negative gradients over the whole node.
    pub fn total_gradient(&self) -> F {
        self.total_gradient
    }

    /// Sum of Hessians over the whole node.
    pub fn total_hessian(&self) -> F {
        self.total_hessian
    }

    /// Number of node rows.
    pub fn total_count(&self) -> usize {
        self.total_count
    }

    /// Number of bins carried by the histogram (one stride for every feature).
    pub fn stride(&self) -> usize {
        self.gradient_sums.len()
    }

    /// Subtract `subtrahend` cell by cell, keeping the totals consistent.
    #[allow(clippy::assign_op_pattern)]
    pub(crate) fn subtract_assign(&mut self, subtrahend: &Self) {
        self.gradient_sums
            .iter_mut()
            .zip(subtrahend.gradient_sums.iter())
            .for_each(|(destination, &value)| *destination = *destination - value);
        self.hessian_sums
            .iter_mut()
            .zip(subtrahend.hessian_sums.iter())
            .for_each(|(destination, &value)| *destination = *destination - value);
        self.counts
            .iter_mut()
            .zip(subtrahend.counts.iter())
            .for_each(|(destination, &value)| *destination = *destination - value);
        self.total_gradient = self.total_gradient - subtrahend.total_gradient;
        self.total_hessian = self.total_hessian - subtrahend.total_hessian;
        self.total_count -= subtrahend.total_count;
    }
}

/// Histograms of every feature for one node.
#[derive(Debug, Clone, PartialEq)]
pub struct SKNodeHistogram<F> {
    pub(crate) features: Vec<SKFeatureHistogram<F>>,
}

impl<F: SKFloat> SKNodeHistogram<F> {
    /// The histogram of `feature_index`.
    pub fn feature(&self, feature_index: usize) -> &SKFeatureHistogram<F> {
        &self.features[feature_index]
    }

    /// Number of features.
    pub fn number_of_features(&self) -> usize {
        self.features.len()
    }
}

/// Accumulate every feature histogram over `rows`, one feature per rayon task.
pub fn sk_accumulate_histogram<F: SKFloat>(
    dataset: &SKBinnedDataset<F>,
    rows: &[usize],
    negative_gradients: &[F],
    hessians: &[F],
) -> SKNodeHistogram<F> {
    let stride = dataset.maximum_bins() + 1;
    let features = (0..dataset.number_of_features())
        .into_par_iter()
        .map(|feature| {
            let mut histogram = SKFeatureHistogram::empty(stride);
            let column = dataset.bins_for_feature(feature);
            rows.iter().for_each(|&row| {
                let bin = usize::from(column[row]);
                histogram.gradient_sums[bin] =
                    histogram.gradient_sums[bin] + negative_gradients[row];
                histogram.hessian_sums[bin] = histogram.hessian_sums[bin] + hessians[row];
                histogram.counts[bin] += 1;
                histogram.total_gradient = histogram.total_gradient + negative_gradients[row];
                histogram.total_hessian = histogram.total_hessian + hessians[row];
                histogram.total_count += 1;
            });
            histogram
        })
        .collect();
    SKNodeHistogram { features }
}
