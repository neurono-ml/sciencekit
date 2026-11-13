//! Per-column donor selection for the nearest-neighbor imputer
//! (spec `nearest-neighbor-imputation`).

use std::cmp::Ordering;

use ndarray::ArrayView1;
use sciencekit_common::SKFloat;

use crate::knn::weights::{SKImputationWeights, sk_weighted_average};

/// The training rows that observed each column, in ascending row order.
///
/// A row that is missing the column is never a donor for it. An all-missing
/// column contributes no donor at all.
pub(super) fn sk_column_donors(missing_mask: &ndarray::ArrayView2<'_, bool>) -> Vec<Vec<usize>> {
    let (rows_count, columns_count) = missing_mask.dim();
    let mut donors = vec![Vec::new(); columns_count];
    for row in 0..rows_count {
        for column in 0..columns_count {
            if !missing_mask[[row, column]] {
                donors[column].push(row);
            }
        }
    }
    donors
}

/// The `nearest_neighbors_count` donors with the smallest finite distance.
///
/// A `NaN` distance means the receiver and the donor share no observed
/// feature, so that donor cannot be compared and is skipped. Ties break by the
/// smallest training row index, and the result is ordered nearest first.
pub(super) fn sk_select_nearest_donors<F: SKFloat>(
    distances: &ArrayView1<'_, F>,
    donors: &[usize],
    nearest_neighbors_count: usize,
) -> Vec<usize> {
    let mut candidates: Vec<(F, usize)> = donors
        .iter()
        .filter_map(|donor| {
            let distance = distances[*donor];
            distance.is_finite().then_some((distance, *donor))
        })
        .collect();
    if candidates.is_empty() || nearest_neighbors_count == 0 {
        return Vec::new();
    }
    let compare = |left: &(F, usize), right: &(F, usize)| {
        left.0
            .partial_cmp(&right.0)
            .unwrap_or(Ordering::Equal)
            .then(left.1.cmp(&right.1))
    };
    let take = nearest_neighbors_count.min(candidates.len());
    if take < candidates.len() {
        candidates.select_nth_unstable_by(take - 1, compare);
        candidates.truncate(take);
    }
    candidates.sort_unstable_by(compare);
    candidates.into_iter().map(|(_, donor)| donor).collect()
}

/// The value that fills a missing cell from its column donors.
///
/// `distances` holds the receiver row's distance to every training row and
/// `training_column` the observed values of the column being filled. When the
/// column observed nothing the cell fills with `0.0` (a retained empty
/// column), when no donor distance is finite it falls back to the observed
/// column mean, and otherwise it averages the selected donors.
pub(super) fn sk_impute_cell<F: SKFloat>(
    distances: &ArrayView1<'_, F>,
    donors: &[usize],
    training_column: &ArrayView1<'_, F>,
    column_mean: F,
    nearest_neighbors_count: usize,
    weights: SKImputationWeights,
) -> F {
    if donors.is_empty() {
        return F::zero();
    }
    let selected = sk_select_nearest_donors(distances, donors, nearest_neighbors_count);
    if selected.is_empty() {
        return column_mean;
    }
    let selected_distances: Vec<F> = selected.iter().map(|donor| distances[*donor]).collect();
    let selected_values: Vec<F> = selected
        .iter()
        .map(|donor| training_column[*donor])
        .collect();
    let donor_weights = weights.weights(&selected_distances);
    sk_weighted_average(&donor_weights, &selected_values)
}
