//! Configuration vocabulary for Isolation Forest.

use sciencekit_common::SKFloat;

/// Default maximum subsample size per isolation tree under [`SKIsolationMaximumSamples::Auto`].
pub const SK_ISOLATION_DEFAULT_MAXIMUM_SAMPLES: usize = 256;

/// Documented fixed score offset applied when [`SKContamination::Auto`] is selected.
pub const SK_ISOLATION_AUTO_OFFSET: f64 = -0.5;

/// Subsample-size policy per isolation tree.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SKIsolationMaximumSamples<F> {
    /// Draw `min(256, row_count)` rows per tree (default).
    Auto,
    /// Draw an explicit number of rows in `1..=row_count`.
    Count(usize),
    /// Draw a fraction in `(0, 1]` of the training rows, floored and clamped to `1..=row_count`.
    Fraction(F),
}

impl<F: SKFloat> SKIsolationMaximumSamples<F> {
    /// Resolves the subsample size against `row_count`.
    pub fn resolved_count(&self, row_count: usize) -> usize {
        if row_count == 0 {
            return 0;
        }
        let resolved = match self {
            Self::Auto => SK_ISOLATION_DEFAULT_MAXIMUM_SAMPLES.min(row_count),
            Self::Count(count) => *count,
            Self::Fraction(fraction) => {
                let scaled = *fraction * F::from(row_count).expect("row count fits scalar");
                scaled.floor().to_usize().unwrap_or(1)
            }
        };
        resolved.clamp(1, row_count)
    }
}

/// Contamination prior governing the decision-function threshold offset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SKContamination<F> {
    /// Use the fixed paper offset [`SK_ISOLATION_AUTO_OFFSET`] (`-0.5`, default).
    Auto,
    /// Set the offset to this quantile in `(0, 0.5]` of the training `score_samples`.
    Fraction(F),
}
