//! The logistic objective primitives: the sigmoid, the softmax and the binary
//! and multinomial loss gradients (spec `logistic-solver`).
//!
//! Every helper is numerically stable in the tails: the sigmoid never
//! overflows, `log_sigmoid` is exact for large magnitudes and the softmax
//! subtracts the row maximum before exponentiating.

use sciencekit_common::SKFloat;

/// The logistic sigmoid `1 / (1 + e^{−z})`, computed without overflow.
pub(crate) fn sk_sigmoid<F: SKFloat>(z: F) -> F {
    if z >= F::zero() {
        F::one() / (F::one() + (-z).exp())
    } else {
        let exponential = z.exp();
        exponential / (F::one() + exponential)
    }
}

/// The log-sigmoid `log(1 / (1 + e^{−z}))`, exact in the tails.
pub(crate) fn sk_log_sigmoid<F: SKFloat>(z: F) -> F {
    if z >= F::zero() {
        -((-z).exp()).ln_1p()
    } else {
        z - z.exp().ln_1p()
    }
}

/// The softmax of `logits`, written into `out` (which must match in length).
pub(crate) fn sk_softmax<F: SKFloat>(logits: &[F], out: &mut [F]) {
    let maximum = logits
        .iter()
        .copied()
        .fold(F::neg_infinity(), |accumulator, value| {
            accumulator.max(value)
        });
    let mut total = F::zero();
    out.iter_mut().zip(logits).for_each(|(entry, &logit)| {
        let exponential = (logit - maximum).exp();
        *entry = exponential;
        total = total + exponential;
    });
    let inverse = F::one() / total;
    out.iter_mut().for_each(|entry| *entry = *entry * inverse);
}

/// The weighted binary logistic loss `c · log(1 + e^{−y·z})` with `y ∈ {−1, 1}`.
#[cfg(test)]
pub(crate) fn sk_binary_loss<F: SKFloat>(linear: F, label: F, weight: F) -> F {
    weight * (-sk_log_sigmoid(label * linear))
}

/// The weighted binary logistic gradient `∂loss/∂z = c · y · (σ(y·z) − 1)`.
#[cfg(test)]
pub(crate) fn sk_binary_gradient<F: SKFloat>(linear: F, label: F, weight: F) -> F {
    weight * label * (sk_sigmoid(label * linear) - F::one())
}

/// The multinomial softmax cross-entropy `−log p_label`.
pub(crate) fn sk_multinomial_loss<F: SKFloat>(logits: &[F], label: usize, weight: F) -> F {
    let maximum = logits
        .iter()
        .copied()
        .fold(F::neg_infinity(), |accumulator, value| {
            accumulator.max(value)
        });
    let log_sum_exp = logits
        .iter()
        .map(|&logit| (logit - maximum).exp())
        .fold(F::zero(), |accumulator, term| accumulator + term)
        .ln();
    weight * (maximum + log_sum_exp - logits[label])
}
