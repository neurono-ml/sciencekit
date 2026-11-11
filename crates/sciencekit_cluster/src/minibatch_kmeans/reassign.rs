//! Deterministic low-count center reassignment (spec `mini-batch-k-means`,
//! decision 4).

use std::cmp::Ordering;

use ndarray::{Array1, Array2};
use sciencekit_common::{SKDataView, SKError, SKFloat};

use super::assign::{compute_center_norms, dense_squared_distance, sparse_squared_distance};

/// Reassigns any center whose count is strictly below `reassignment_ratio ×
/// mean_count` to the batch row with the largest squared distance to its
/// assigned center (breaking ties by smallest row index), resetting its count
/// to `1`. Returns the number of reassigned centers.
pub(crate) fn sk_reassign_low_count_centers<F: SKFloat>(
    centers: &mut Array2<F>,
    counts: &mut Array1<usize>,
    features: SKDataView<'_, F>,
    assignments: &Array1<usize>,
    reassignment_ratio: F,
) -> Result<usize, SKError> {
    let number_of_clusters = centers.nrows();
    if number_of_clusters == 0 || reassignment_ratio <= F::zero() {
        return Ok(0);
    }

    let total_count: usize = counts.iter().sum();
    let mean_count = F::from(total_count).expect("count sum fits scalar")
        / F::from(number_of_clusters).expect("cluster count fits scalar");
    let threshold = reassignment_ratio * mean_count;

    let candidate_centers: Vec<usize> = counts
        .iter()
        .enumerate()
        .filter_map(|(cluster, &count)| {
            let scalar_count = F::from(count).expect("cluster count fits scalar");
            (scalar_count < threshold).then_some(cluster)
        })
        .collect();
    if candidate_centers.is_empty() {
        return Ok(0);
    }

    let mut ranked_rows: Vec<(usize, F)> = match &features {
        SKDataView::Dense(dense) => dense
            .rows()
            .into_iter()
            .zip(assignments.iter())
            .enumerate()
            .map(|(row_index, (row, &cluster))| {
                let distance = dense_squared_distance(row, centers.row(cluster));
                (row_index, distance)
            })
            .collect(),
        SKDataView::Sparse(sparse) => {
            let center_norms = compute_center_norms(&centers.view());
            sparse
                .outer_iterator()
                .zip(assignments.iter())
                .enumerate()
                .map(|(row_index, (stored_row, &cluster))| {
                    let row_norm = stored_row
                        .data()
                        .iter()
                        .fold(F::zero(), |accumulator, &value| accumulator + value * value);
                    let distance = sparse_squared_distance(
                        stored_row,
                        row_norm,
                        centers.row(cluster),
                        center_norms[cluster],
                    );
                    (row_index, distance)
                })
                .collect()
        }
        _ => {
            return Err(SKError::UnsupportedRepresentation {
                representation: features.representation(),
                suggestion: "convert the input to a supported dense or CSR view",
            });
        }
    };

    ranked_rows.sort_by(
        |&(left_index, left_distance), &(right_index, right_distance)| {
            right_distance
                .partial_cmp(&left_distance)
                .unwrap_or(Ordering::Equal)
                .then_with(|| left_index.cmp(&right_index))
        },
    );

    let pairs: Vec<(usize, usize)> = candidate_centers
        .into_iter()
        .zip(ranked_rows.into_iter().map(|(row_index, _)| row_index))
        .collect();
    let reassigned_count = pairs.len();

    match features {
        SKDataView::Dense(dense) => {
            pairs.into_iter().for_each(|(cluster, row_index)| {
                centers.row_mut(cluster).assign(&dense.row(row_index));
                counts[cluster] = 1;
            });
        }
        SKDataView::Sparse(sparse) => {
            pairs.into_iter().for_each(|(cluster, row_index)| {
                let mut center_row = centers.row_mut(cluster);
                center_row.fill(F::zero());
                if let Some(stored) = sparse.outer_view(row_index) {
                    stored.iter().for_each(|(column, &value)| {
                        center_row[column] = value;
                    });
                }
                counts[cluster] = 1;
            });
        }
        _ => unreachable!("representation already checked above"),
    }

    Ok(reassigned_count)
}
