//! The row-subsampling policy for bootstrap ensembles.

use sciencekit_common::SKFloat;

/// How many training rows each tree draws for its sample.
#[derive(Debug, Clone, PartialEq)]
pub enum SKMaximumSamples<F> {
    /// Draw as many samples as there are training rows (default).
    All,
    /// Draw an explicit number of samples, clamped to `[1, row_count]`.
    Count(usize),
    /// Draw this fraction of the training rows, floored and clamped to
    /// `[1, row_count]`.
    Fraction(F),
}

impl<F: SKFloat> SKMaximumSamples<F> {
    /// Resolve the target sample size against `row_count`.
    pub fn resolved_count(&self, row_count: usize) -> usize {
        if row_count == 0 {
            return 0;
        }
        let resolved = match self {
            SKMaximumSamples::All => row_count,
            SKMaximumSamples::Count(count) => *count,
            SKMaximumSamples::Fraction(fraction) => {
                let scaled = *fraction * F::from(row_count).expect("row count fits in scalar");
                scaled.to_usize().unwrap_or(1)
            }
        };
        resolved.clamp(1, row_count)
    }
}
