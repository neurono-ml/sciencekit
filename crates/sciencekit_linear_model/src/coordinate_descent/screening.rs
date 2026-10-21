//! Gap-safe screening with a KKT reactivation guarantee (spec
//! `coordinate-descent-solver`, Decision 5).
//!
//! The pure-L1 (lasso) objective has no closed form but a firm primal–dual
//! bound: with `z` any dual-feasible point (`‖Xᵀz‖∞ ≤ α`) and `z*` the dual
//! optimum, the primal–dual gap `G` bounds the distance `‖z* − z‖₂ ≤ √(2G/n)`.
//! A coordinate `j` is then provably zero when `|Xⱼᵀz| + ‖Xⱼ‖₂·√(2G/n) < α`,
//! because `|Xⱼᵀz*| < α` contradicts complementary slackness for `βⱼ ≠ 0`.
//!
//! Screening can only remove work, never change the answer: after the screened
//! active set converges, [`sk_kkt_violations`] checks **every** coordinate's
//! full gradient and the solver reactivates any violator, so the final
//! coefficients solve the unscreened problem by construction.

use sciencekit_common::SKFloat;

/// The gap-safe screening radius `√(2·gap/n)` around the dual-feasible point.
pub(crate) fn sk_gap_safe_radius<F: SKFloat>(duality_gap: F, row_count: usize) -> F {
    let row_count = F::from(row_count).expect("row counts fit in the scalar");
    let two = F::one() + F::one();
    let non_negative_gap = if duality_gap > F::zero() {
        duality_gap
    } else {
        F::zero()
    };
    (two * non_negative_gap / row_count).sqrt()
}

/// The scaling that turns the raw residual into a dual-feasible point:
/// `z = −r / (n · scale)`, with `scale = max(1, ‖Xᵀr/n‖∞ / α)`.
pub(crate) fn sk_dual_feasible_scale<F: SKFloat>(
    gradient_infinity_norm: F,
    l1_threshold: F,
) -> F {
    if l1_threshold <= F::zero() {
        return F::one();
    }
    let ratio = gradient_infinity_norm / l1_threshold;
    if ratio > F::one() { ratio } else { F::one() }
}

/// Whether a coordinate is *provably* zero by the gap-safe test.
pub(crate) fn sk_is_safely_zero<F: SKFloat>(
    centered_gradient_abs: F,
    column_norm: F,
    radius: F,
    l1_threshold: F,
) -> bool {
    centered_gradient_abs + column_norm * radius < l1_threshold
}

/// The inactive coordinates whose full gradient violates the KKT conditions and
/// must be reactivated. `gradient[j]` is `xⱼᵀr / n` at the current solution.
pub(crate) fn sk_kkt_violations<F: SKFloat>(
    gradient: &[F],
    active: &[bool],
    l1_threshold: F,
    tolerance: F,
) -> Vec<usize> {
    gradient
        .iter()
        .enumerate()
        .filter(|(index, value)| {
            !active[*index] && value.abs() > l1_threshold + tolerance
        })
        .map(|(index, _)| index)
        .collect()
}
