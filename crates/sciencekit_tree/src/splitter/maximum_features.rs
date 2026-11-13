//! The feature-subsampling policy and its resolved feature count.

use sciencekit_common::SKFloat;

/// The feature-subsampling policy applied at every split.
#[derive(Debug, Clone, PartialEq)]
pub enum SKMaximumFeatures<F> {
    /// Consider every feature (default).
    All,
    /// Consider exactly this many features.
    Count(usize),
    /// Consider this fraction of the features.
    Fraction(F),
    /// Consider the square root of the feature count.
    Sqrt,
    /// Consider the base-2 logarithm of the feature count.
    Log2,
}

impl<F: SKFloat> SKMaximumFeatures<F> {
    /// Resolve the policy against the design's feature count.
    ///
    /// `All` and `Count` cannot exceed the available features; `Fraction`
    /// truncates like the reference implementation; `Sqrt` and `Log2` floor
    /// their result. Every policy considers at least one feature.
    pub fn resolved_count(&self, feature_count: usize) -> usize {
        if feature_count == 0 {
            return 0;
        }
        let resolved = match self {
            SKMaximumFeatures::All => feature_count,
            SKMaximumFeatures::Count(count) => *count,
            SKMaximumFeatures::Fraction(fraction) => {
                let scaled = *fraction * F::from(feature_count).expect("count fits");
                scaled.to_usize().unwrap_or(1)
            }
            SKMaximumFeatures::Sqrt => (F::from(feature_count).expect("count fits").sqrt())
                .to_usize()
                .unwrap_or(1),
            SKMaximumFeatures::Log2 => (F::from(feature_count).expect("count fits").log2())
                .to_usize()
                .unwrap_or(1),
        };
        resolved.clamp(1, feature_count)
    }
}
