//! The classifier loss catalogue (spec `stochastic-gradient-solver`,
//! decision 2).
//!
//! Targets are binary `±1`, one per one-vs-rest head. Every loss matches
//! scikit-learn's definition on the margin `y·z`.

use sciencekit_common::SKFloat;

use super::definition::SKSGDLoss;
use super::definition::private::SKSGDLossSealed;

/// The loss of a binary one-vs-rest head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKSGDClassifierLoss {
    /// The hinge loss `max(0, 1 − y·z)`, the linear SVM objective.
    Hinge,
    /// The logistic loss `log(1 + e^{−y·z})`, calibrated.
    LogLoss,
    /// The modified Huber loss, quadratic near the boundary and linear beyond.
    ModifiedHuber,
    /// The squared hinge loss `max(0, 1 − y·z)²`.
    SquaredHinge,
    /// The perceptron loss `max(0, −y·z)`.
    Perceptron,
}

impl<F: SKFloat> SKSGDLossSealed<F> for SKSGDClassifierLoss {}

impl<F: SKFloat> SKSGDLoss<F> for SKSGDClassifierLoss {
    fn value(&self, prediction: F, target: F) -> F {
        let margin = target * prediction;
        match self {
            Self::Hinge => (F::one() - margin).max(F::zero()),
            Self::LogLoss => -sk_log_sigmoid(margin),
            Self::ModifiedHuber => {
                if margin >= -F::one() {
                    (F::one() - margin).max(F::zero()).powi(2)
                } else {
                    -F::from(4).expect("four is representable") * margin
                }
            }
            Self::SquaredHinge => (F::one() - margin).max(F::zero()).powi(2),
            Self::Perceptron => (-margin).max(F::zero()),
        }
    }

    fn slope(&self, prediction: F, target: F) -> F {
        let margin = target * prediction;
        match self {
            Self::Hinge => {
                if margin < F::one() {
                    -target
                } else {
                    F::zero()
                }
            }
            Self::LogLoss => target * (sk_sigmoid(margin) - F::one()),
            Self::ModifiedHuber => {
                if margin >= -F::one() {
                    if margin < F::one() {
                        -F::from(2).expect("two is representable") * target * (F::one() - margin)
                    } else {
                        F::zero()
                    }
                } else {
                    -F::from(4).expect("four is representable") * target
                }
            }
            Self::SquaredHinge => {
                if margin < F::one() {
                    -F::from(2).expect("two is representable") * target * (F::one() - margin)
                } else {
                    F::zero()
                }
            }
            Self::Perceptron => {
                if margin <= F::zero() {
                    -target
                } else {
                    F::zero()
                }
            }
        }
    }

    fn is_probabilistic(&self) -> bool {
        matches!(self, Self::LogLoss | Self::ModifiedHuber)
    }

    fn probability(&self, prediction: F) -> F {
        match self {
            Self::LogLoss => sk_sigmoid(prediction),
            Self::ModifiedHuber => {
                let probability = (prediction + F::one()) / F::from(2).expect("two");
                probability.max(F::zero()).min(F::one())
            }
            _ => F::nan(),
        }
    }
}

/// The logistic sigmoid `1 / (1 + e^{−z})`, computed without overflow.
fn sk_sigmoid<F: SKFloat>(value: F) -> F {
    if value >= F::zero() {
        F::one() / (F::one() + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (F::one() + exponential)
    }
}

/// The log-sigmoid `log(1 / (1 + e^{−z}))`, exact in the tails.
fn sk_log_sigmoid<F: SKFloat>(value: F) -> F {
    if value >= F::zero() {
        -((-value).exp()).ln_1p()
    } else {
        value - value.exp().ln_1p()
    }
}
