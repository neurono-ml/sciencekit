//! Aggregation of per-class scores into macro, micro, weighted or binary scores.

use sciencekit_common::{SKError, SKFloat};

use super::confusion::SKConfusionMatrix;
use super::precision_recall_f1::{SKClassScores, sk_ratio};

/// How per-class classification scores are aggregated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SKAveraging {
    /// The unweighted mean over every resolved class.
    Macro,
    /// The aggregate over the total true positives and observation counts.
    Micro,
    /// The mean weighted by each class's true support.
    Weighted,
    /// The score of one chosen class of a binary problem.
    Binary {
        /// The label whose per-class score is reported.
        positive_label: i64,
    },
}

/// The aggregated precision, recall and F1 of a classification evaluation.
#[derive(Debug, Clone, PartialEq)]
pub struct SKAggregatedScores<F> {
    precision: F,
    recall: F,
    f1: F,
}

impl<F: SKFloat> SKAggregatedScores<F> {
    /// The aggregated precision.
    pub fn precision(&self) -> F {
        self.precision
    }

    /// The aggregated recall.
    pub fn recall(&self) -> F {
        self.recall
    }

    /// The aggregated F1.
    pub fn f1(&self) -> F {
        self.f1
    }
}

/// Aggregate per-class scores with the configured averaging policy.
pub fn sk_aggregate_class_scores<F: SKFloat>(
    matrix: &SKConfusionMatrix,
    scores: &SKClassScores<F>,
    averaging: SKAveraging,
) -> Result<SKAggregatedScores<F>, SKError> {
    match averaging {
        SKAveraging::Macro => Ok(SKAggregatedScores {
            precision: sk_mean(scores.precision()),
            recall: sk_mean(scores.recall()),
            f1: sk_mean(scores.f1()),
        }),
        SKAveraging::Micro => {
            let classes = matrix.number_of_classes();
            let true_positives: usize = (0..classes).map(|index| matrix.true_positive(index)).sum();
            let observations = matrix.number_of_observations();
            let precision = sk_ratio::<F>(true_positives, observations);
            let recall = precision;
            let denominator = precision + recall;
            let f1 = if denominator.is_zero() {
                F::zero()
            } else {
                (F::one() + F::one()) * precision * recall / denominator
            };
            Ok(SKAggregatedScores {
                precision,
                recall,
                f1,
            })
        }
        SKAveraging::Weighted => {
            let total_support: usize = (0..matrix.number_of_classes())
                .map(|index| matrix.support(index))
                .sum();
            Ok(SKAggregatedScores {
                precision: sk_weighted_mean(scores.precision(), matrix, total_support),
                recall: sk_weighted_mean(scores.recall(), matrix, total_support),
                f1: sk_weighted_mean(scores.f1(), matrix, total_support),
            })
        }
        SKAveraging::Binary { positive_label } => {
            let index = matrix
                .labels()
                .binary_search(&positive_label)
                .map_err(|_| SKError::UnknownLabel {
                    label: positive_label,
                })?;
            Ok(SKAggregatedScores {
                precision: scores.precision()[index],
                recall: scores.recall()[index],
                f1: scores.f1()[index],
            })
        }
    }
}

/// The unweighted mean of a non-empty slice.
fn sk_mean<F: SKFloat>(values: &[F]) -> F {
    let total = values.iter().fold(F::zero(), |sum, value| sum + *value);
    let count = num_traits::cast::<usize, F>(values.len()).expect("counts fit in the scalar");
    total / count
}

/// The support-weighted mean of per-class scores.
fn sk_weighted_mean<F: SKFloat>(
    values: &[F],
    matrix: &SKConfusionMatrix,
    total_support: usize,
) -> F {
    let weighted = values
        .iter()
        .enumerate()
        .fold(F::zero(), |sum, (index, value)| {
            let support = num_traits::cast::<usize, F>(matrix.support(index))
                .expect("counts fit in the scalar");
            sum + *value * support
        });
    let total = num_traits::cast::<usize, F>(total_support).expect("counts fit in the scalar");
    weighted / total
}
