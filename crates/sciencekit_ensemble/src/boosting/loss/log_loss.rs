//! Binary (sigmoid) and multiclass (softmax) log-loss implementations for
//! classification gradient boosting.

use ndarray::Array2;
use sciencekit_common::{SKError, SKFloat, sk_validate_hyperparameter};

use super::contract::{SKGradientLoss, sk_target_mean};

/// Binary cross-entropy loss with logistic sigmoid link.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SKBinaryLogLoss;

impl SKBinaryLogLoss {
    /// Construct the binary log-loss.
    pub fn new() -> Self {
        Self
    }

    ///Numerically stable logistic sigmoid `1 / (1 + exp(-raw_score))`.
    pub fn sigmoid<F: SKFloat>(raw_score: F) -> F {
        if raw_score >= F::zero() {
            F::one() / (F::one() + (-raw_score).exp())
        } else {
            let exponential = raw_score.exp();
            exponential / (F::one() + exponential)
        }
    }
}

impl<F: SKFloat> SKGradientLoss<F> for SKBinaryLogLoss {
    fn initial_prediction(&self, targets: &[F]) -> F {
        if targets.is_empty() {
            return F::zero();
        }
        self.link(sk_target_mean(targets))
    }

    fn negative_gradient(&self, target: F, raw_prediction: F) -> F {
        target - Self::sigmoid(raw_prediction)
    }

    fn hessian(&self, _target: F, raw_prediction: F) -> F {
        let probability = Self::sigmoid(raw_prediction);
        probability * (F::one() - probability)
    }

    fn link(&self, probability: F) -> F {
        let epsilon = F::epsilon();
        let clamped = probability.max(epsilon).min(F::one() - epsilon);
        (clamped / (F::one() - clamped)).ln()
    }

    fn inverse_link(&self, raw_prediction: F) -> F {
        Self::sigmoid(raw_prediction)
    }

    fn loss(&self, raw_prediction: F, target: F) -> F {
        raw_prediction.max(F::zero()) - raw_prediction * target
            + (F::one() + (-raw_prediction.abs()).exp()).ln()
    }
}

/// Multiclass cross-entropy loss with softmax link and diagonal Hessian.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SKMulticlassLogLoss {
    number_of_classes: usize,
}

impl SKMulticlassLogLoss {
    /// Start building a multiclass log-loss.
    pub fn build() -> SKMulticlassLogLossBuilder {
        SKMulticlassLogLossBuilder::new()
    }

    /// The number of canonical classes `C`.
    pub fn number_of_classes(&self) -> usize {
        self.number_of_classes
    }

    /// Compute the `C` initial class log-priors from canonical `class_indices`.
    pub fn initial_predictions<F: SKFloat>(&self, class_indices: &[usize]) -> Vec<F> {
        if class_indices.is_empty() {
            return vec![F::zero(); self.number_of_classes];
        }
        let mut counts = vec![0usize; self.number_of_classes];
        class_indices
            .iter()
            .for_each(|&class_index| counts[class_index] += 1);
        let total = F::from(class_indices.len()).expect("row count fits in scalar");
        let epsilon = F::epsilon();
        counts
            .into_iter()
            .map(|count| {
                let prior = F::from(count).expect("count fits") / total;
                prior.max(epsilon).min(F::one() - epsilon).ln()
            })
            .collect()
    }

    /// Map a length-`C` raw score slice to normalized softmax probabilities.
    pub fn softmax<F: SKFloat>(&self, raw_scores: &[F]) -> Vec<F> {
        if raw_scores.is_empty() {
            return Vec::new();
        }
        let maximum = raw_scores
            .iter()
            .copied()
            .fold(F::neg_infinity(), |best, score| best.max(score));
        let exponentials: Vec<F> = raw_scores
            .iter()
            .map(|&score| (score - maximum).exp())
            .collect();
        let normalizer = exponentials
            .iter()
            .fold(F::zero(), |sum, &value| sum + value);
        exponentials
            .into_iter()
            .map(|value| value / normalizer)
            .collect()
    }

    /// Produce `C` per-class negative-gradient vectors and `C` diagonal-Hessian
    /// vectors over `raw_scores` of shape `(rows, C)`.
    pub fn stage_gradients_and_hessians<F: SKFloat>(
        &self,
        class_indices: &[usize],
        raw_scores: &Array2<F>,
    ) -> (Vec<Vec<F>>, Vec<Vec<F>>) {
        let rows_count = class_indices.len();
        let mut negative_gradients = vec![Vec::with_capacity(rows_count); self.number_of_classes];
        let mut hessians = vec![Vec::with_capacity(rows_count); self.number_of_classes];
        class_indices
            .iter()
            .zip(raw_scores.rows())
            .for_each(|(&true_class, score_row)| {
                let slice: Vec<F> = score_row.iter().copied().collect();
                let probabilities = self.softmax(&slice);
                probabilities
                    .into_iter()
                    .enumerate()
                    .for_each(|(class_index, probability)| {
                        let indicator = if class_index == true_class {
                            F::one()
                        } else {
                            F::zero()
                        };
                        negative_gradients[class_index].push(indicator - probability);
                        hessians[class_index].push(probability * (F::one() - probability));
                    });
            });
        (negative_gradients, hessians)
    }

    /// Compute the multiclass log-loss for one row with `true_class`.
    pub fn loss_for_row<F: SKFloat>(&self, raw_scores: &[F], true_class: usize) -> F {
        let maximum = raw_scores
            .iter()
            .copied()
            .fold(F::neg_infinity(), |best, score| best.max(score));
        let sum_exponentials = raw_scores
            .iter()
            .fold(F::zero(), |sum, &score| sum + (score - maximum).exp());
        maximum + sum_exponentials.ln() - raw_scores[true_class]
    }
}

/// Builder for [`SKMulticlassLogLoss`].
#[derive(Debug, Clone, Copy, Default)]
pub struct SKMulticlassLogLossBuilder {
    number_of_classes: usize,
}

impl SKMulticlassLogLossBuilder {
    /// Create a builder (must set `number_of_classes >= 2`).
    pub fn new() -> Self {
        Self {
            number_of_classes: 0,
        }
    }

    /// Set the number of canonical classes (`>= 2`).
    pub fn number_of_classes(mut self, number_of_classes: usize) -> Self {
        self.number_of_classes = number_of_classes;
        self
    }

    /// Validate and construct [`SKMulticlassLogLoss`].
    pub fn build(self) -> Result<SKMulticlassLogLoss, SKError> {
        sk_validate_hyperparameter(
            "number_of_classes",
            self.number_of_classes >= 2,
            "multiclass log-loss requires at least two classes",
        )?;
        Ok(SKMulticlassLogLoss {
            number_of_classes: self.number_of_classes,
        })
    }
}
