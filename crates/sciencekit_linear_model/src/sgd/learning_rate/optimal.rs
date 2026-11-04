//! The optimal-schedule offset estimate (spec `stochastic-gradient-solver`,
//! decision 4).
//!
//! scikit-learn fixes a constant `t₀`; the substrate estimates it from the
//! initial loss scale instead, so the schedule is deterministic but not
//! bit-identical to scikit-learn.

#[cfg(test)]
use sciencekit_common::SKFloat;

/// The offset `t₀ = max(1, 1 / (alpha · s))` of the optimal schedule.
///
/// `s` is the mean absolute slope over the first batch. A non-positive `s` (or
/// a non-positive `alpha`, which validation rejects) falls back to one. The
/// estimators estimate the offset on their own target layout, so this helper is
/// exercised by the substrate's tests.
#[cfg(test)]
pub(crate) fn sk_optimal_offset<F: SKFloat>(alpha: F, mean_absolute_slope: F) -> F {
    if alpha <= F::zero() || mean_absolute_slope <= F::zero() {
        return F::one();
    }
    (F::one() / (alpha * mean_absolute_slope)).max(F::one())
}
