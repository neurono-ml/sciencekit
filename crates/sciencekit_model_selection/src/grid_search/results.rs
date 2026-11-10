//! Cross-validated score results and best-candidate selection
//! (spec `grid-search`, requirement `Score aggregation and best selection`).

use std::cmp::Ordering;

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sciencekit_common::{SKError, SKFloat};

/// The immutable outcome of a grid search fit.
///
/// Fold scores are stored by candidate and fold index, so the matrix never
/// depends on completion order. Mean and population standard deviation are
/// aggregated per candidate, and `best_index` is the highest mean with ties
/// broken by the earlier candidate index.
#[derive(Debug, Clone, PartialEq)]
pub struct SKGridSearchResults<F: SKFloat> {
    fold_scores: Array2<F>,
    mean_scores: Array1<F>,
    standard_deviations: Array1<F>,
    best_index: usize,
}

impl<F: SKFloat> SKGridSearchResults<F> {
    /// Aggregates a `(candidates × folds)` score matrix.
    pub(crate) fn from_scores(scores: Array2<F>) -> Result<Self, SKError> {
        if scores.nrows() == 0 || scores.ncols() == 0 {
            return Err(SKError::EmptyInput {
                operation: "grid_search",
            });
        }
        let (mean_scores, standard_deviations) = sk_aggregate_rows(scores.view());
        let best_index = sk_best_candidate_index(mean_scores.view());
        Ok(Self {
            fold_scores: scores,
            mean_scores,
            standard_deviations,
            best_index,
        })
    }

    /// The number of evaluated candidates.
    pub fn number_of_candidates(&self) -> usize {
        self.fold_scores.nrows()
    }

    /// The number of folds each candidate was evaluated on.
    pub fn number_of_folds(&self) -> usize {
        self.fold_scores.ncols()
    }

    /// The raw `(candidates × folds)` score matrix.
    pub fn fold_scores(&self) -> ArrayView2<'_, F> {
        self.fold_scores.view()
    }

    /// The mean score of every candidate.
    pub fn mean_scores(&self) -> ArrayView1<'_, F> {
        self.mean_scores.view()
    }

    /// The population standard deviation of every candidate.
    pub fn standard_deviations(&self) -> ArrayView1<'_, F> {
        self.standard_deviations.view()
    }

    /// The winning candidate index.
    pub fn best_index(&self) -> usize {
        self.best_index
    }

    /// The winning candidate's mean score.
    pub fn best_score(&self) -> F {
        self.mean_scores[self.best_index]
    }

    /// Candidate indices ranked by descending mean, ties by ascending index.
    pub fn ranked_candidate_indices(&self) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.number_of_candidates()).collect();
        indices.sort_by(|&left, &right| {
            sk_compare_scores(self.mean_scores[right], self.mean_scores[left])
                .then(left.cmp(&right))
        });
        indices
    }
}

fn sk_compare_scores<F: SKFloat>(left: F, right: F) -> Ordering {
    left.partial_cmp(&right).unwrap_or(Ordering::Equal)
}

fn sk_aggregate_rows<F: SKFloat>(scores: ArrayView2<'_, F>) -> (Array1<F>, Array1<F>) {
    let number_of_folds = scores.ncols();
    let count = F::from(number_of_folds).expect("the fold count fits the scalar");
    let mut means = Array1::zeros(scores.nrows());
    let mut deviations = Array1::zeros(scores.nrows());
    for (candidate, row) in scores.rows().into_iter().enumerate() {
        let mut total = F::zero();
        for value in row.iter() {
            total = total + *value;
        }
        let mean = total / count;
        let mut squared = F::zero();
        for value in row.iter() {
            let difference = *value - mean;
            squared = squared + difference * difference;
        }
        means[candidate] = mean;
        deviations[candidate] = (squared / count).sqrt();
    }
    (means, deviations)
}

fn sk_best_candidate_index<F: SKFloat>(means: ArrayView1<'_, F>) -> usize {
    let mut best = 0;
    for candidate in 1..means.len() {
        if sk_compare_scores(means[candidate], means[best]) == Ordering::Greater {
            best = candidate;
        }
    }
    best
}
