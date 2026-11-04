//! The penalty selection and the proximal update (spec
//! `stochastic-gradient-solver`, decision 3).

use ndarray::{ArrayViewMut1, azip};
use sciencekit_common::SKFloat;

/// The regularization penalty of a stochastic-gradient fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum SKSGDPenalty {
    /// The L2 (ridge) penalty `0.5·alpha·‖w‖₂²`.
    #[default]
    L2,
    /// The L1 (lasso) penalty `alpha·‖w‖₁`, promoting exact zeros.
    L1,
    /// The mixed L1/L2 penalty, controlled by `l1_ratio`.
    ElasticNet,
    /// No penalty.
    None,
}

impl SKSGDPenalty {
    /// The L1 coefficient applied by this penalty.
    ///
    /// `l1_ratio` is consulted only under [`SKSGDPenalty::ElasticNet`].
    pub(crate) fn l1_coefficient<F: SKFloat>(self, alpha: F, l1_ratio: F) -> F {
        match self {
            Self::L1 => alpha,
            Self::ElasticNet => alpha * l1_ratio,
            Self::L2 | Self::None => F::zero(),
        }
    }

    /// The L2 coefficient applied by this penalty.
    pub(crate) fn l2_coefficient<F: SKFloat>(self, alpha: F, l1_ratio: F) -> F {
        match self {
            Self::L2 => alpha,
            Self::ElasticNet => alpha * (F::one() - l1_ratio),
            Self::L1 | Self::None => F::zero(),
        }
    }
}

/// The soft-threshold operator `sign(v)·max(0, |v| − threshold)`.
pub(crate) fn sk_soft_threshold<F: SKFloat>(value: F, threshold: F) -> F {
    if value > threshold {
        value - threshold
    } else if value < -threshold {
        value + threshold
    } else {
        F::zero()
    }
}

/// Apply the proximal penalty after the gradient step.
///
/// The L2 part decays the weight and the L1 part soft-thresholds it, both
/// scaled by the learning rate. Only the weights are passed in: the intercept
/// is never penalized.
pub(crate) fn sk_apply_proximal_penalty<F: SKFloat>(
    mut weights: ArrayViewMut1<F>,
    learning_rate: F,
    l1_coefficient: F,
    l2_coefficient: F,
) {
    let l1_threshold = learning_rate * l1_coefficient;
    let l2_decay = learning_rate * l2_coefficient;
    azip!((weight in &mut weights) {
        let decayed = *weight - l2_decay * *weight;
        *weight = sk_soft_threshold(decayed, l1_threshold);
    });
}
