//! The pass-count policy an estimator declares for a streaming fit (spec
//! `streaming-source-provider`).

/// How many streaming passes an estimator may take.
///
/// The estimator declares its class; the provider-based entry points enforce
/// the rule. An accumulating estimator that re-read its source would double
/// count its statistics, so it takes exactly one pass regardless of budget.
/// An iterative (stochastic) estimator may take up to its epoch budget from a
/// reopenable provider, and one pass from a one-shot provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKStreamingPassPolicy {
    /// An accumulating estimator takes exactly one pass.
    SinglePass,
    /// An iterative estimator takes up to this many passes from a reopenable
    /// provider.
    Iterative {
        /// The epoch budget of the estimator's fit.
        maximum_number_of_passes: usize,
    },
}

impl SKStreamingPassPolicy {
    /// The passes to attempt for a provider's reopenability.
    pub(super) fn resolve_pass_count(self, is_reopenable: bool) -> usize {
        match self {
            SKStreamingPassPolicy::SinglePass => 1,
            SKStreamingPassPolicy::Iterative {
                maximum_number_of_passes,
            } => {
                if is_reopenable {
                    maximum_number_of_passes.max(1)
                } else {
                    1
                }
            }
        }
    }
}
