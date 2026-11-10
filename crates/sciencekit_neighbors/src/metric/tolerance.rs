//! Conservative pruning comparisons for spatial indexes.
//!
//! A lower bound is computed in floating point, so a node whose *true* bound
//! equals the current limit can round to a value a few ulps above it. Pruning on
//! `bound > limit` would then discard a row that is exactly at the radius (or
//! exactly tied with the worst neighbour), breaking the inclusive-radius and
//! deterministic-tie guarantees. The helpers here add a small relative
//! tolerance, so a bound must exceed the limit by more than rounding noise
//! before the subtree is skipped. Over-exploring costs a few node visits;
//! under-exploring would return a wrong answer.

use sciencekit_common::SKFloat;

/// Number of ulps of headroom granted to the bound before pruning.
const SK_BOUND_TOLERANCE_ULPS: u32 = 8;

/// Reports whether `bound` is safely beyond `limit` (so the node can be pruned).
///
/// The comparison is conservative: a bound within a few ulps of `limit` keeps
/// the node on the stack.
pub(crate) fn sk_lower_bound_exceeds<F: SKFloat>(bound: F, limit: F) -> bool {
    let scale = limit.abs().max(F::one());
    let multiplier = F::from(SK_BOUND_TOLERANCE_ULPS).expect("the ulp count fits in the scalar");
    let tolerance = F::epsilon() * multiplier * scale;
    bound > limit + tolerance
}
