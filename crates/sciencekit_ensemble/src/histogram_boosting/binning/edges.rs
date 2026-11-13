//! Deterministic quantile bin edges and the dedicated missing bin (spec
//! `histogram-gradient-boosting`, design decision 2).

use sciencekit_common::SKFloat;

/// Maximum number of values sampled per column when deriving quantile edges.
const SK_BINNING_SAMPLE_LIMIT: usize = 200_000;

/// Quantile binning of one numeric feature.
///
/// The stored interior `edges` split the observed numeric range into
/// `edges.len() + 1` bins. A dedicated bin, [`SKNumericBinning::missing_bin`],
/// carries missing values and is never produced from a finite value.
#[derive(Debug, Clone, PartialEq)]
pub struct SKNumericBinning<F> {
    pub(super) edges: Vec<F>,
    pub(super) missing_bin: u8,
}

impl<F: SKFloat> SKNumericBinning<F> {
    /// Derive at most `maximum_bins` numeric bins from the finite observed
    /// `values`; `NaN` values are the missing marker and are ignored.
    ///
    /// The derivation is deterministic: values are sorted, deterministically
    /// decimated when the column exceeds [`SK_BINNING_SAMPLE_LIMIT`], and the
    /// interior edges are the equally spaced quantiles of the sample.
    pub fn derive(values: impl Iterator<Item = F>, maximum_bins: usize) -> Self {
        let maximum_bins = maximum_bins.clamp(1, 255);
        let mut finite_values: Vec<F> = values.filter(|value| value.is_finite()).collect();
        let sample_stride = finite_values.len().div_ceil(SK_BINNING_SAMPLE_LIMIT).max(1);
        if sample_stride > 1 {
            finite_values = finite_values.into_iter().step_by(sample_stride).collect();
        }
        finite_values.sort_by(|left, right| left.partial_cmp(right).expect("finite values"));
        let mut edges: Vec<F> = (1..maximum_bins)
            .filter_map(|bin| {
                let position = bin * finite_values.len() / maximum_bins;
                finite_values.get(position).copied()
            })
            .collect();
        edges.dedup();
        Self {
            edges,
            missing_bin: maximum_bins as u8,
        }
    }

    /// The interior quantile edges, ascending.
    pub fn edges(&self) -> &[F] {
        &self.edges
    }

    /// The bin reserved for missing values (`maximum_bins`).
    pub fn missing_bin(&self) -> u8 {
        self.missing_bin
    }

    /// Number of numeric bins (`edges.len() + 1`), excluding the missing bin.
    pub fn number_of_bins(&self) -> usize {
        self.edges.len() + 1
    }

    /// Assign `value` to a bin; `NaN` maps to [`Self::missing_bin`].
    pub fn bin(&self, value: F) -> u8 {
        if value.is_nan() {
            return self.missing_bin;
        }
        let boundary = self.edges.partition_point(|edge| *edge <= value);
        (boundary as u8).min(self.number_of_bins() as u8 - 1)
    }
}
