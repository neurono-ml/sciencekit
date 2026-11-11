//! Per-center learning-rate updates and count bookkeeping (spec
//! `mini-batch-k-means`, decision 3).
//!
//! Sequential application of `c <- c + (x - c) / (count + 1); count += 1` over
//! `m` batch rows assigned to center `c` starting from count `n` reduces
//! algebraically to `c_new = (n / (n + m)) * c_old + (1 / (n + m)) * sum(x)`
//! with `count_new = n + m`. This exact closed form lets CSR batches touch only
//! their stored entries (`O(k·d + nnz)`) without densifying.

use ndarray::{Array1, Array2, azip};
use sciencekit_common::{SKDataView, SKError, SKFloat};

use super::assign::dense_squared_distance;

/// Updates `centers` and `counts` from one assigned batch and returns the
/// largest Euclidean center shift `max_c ‖c_new − c_old‖₂`.
pub(crate) fn sk_update_centers_and_counts<F: SKFloat>(
    centers: &mut Array2<F>,
    counts: &mut Array1<usize>,
    features: SKDataView<'_, F>,
    assignments: &Array1<usize>,
) -> Result<F, SKError> {
    let number_of_clusters = centers.nrows();
    let number_of_features = centers.ncols();
    let previous_centers = centers.clone();

    let mut batch_counts = vec![0_usize; number_of_clusters];
    assignments
        .iter()
        .for_each(|&cluster| batch_counts[cluster] += 1);

    match features {
        SKDataView::Dense(dense) => {
            let mut batch_sums = Array2::<F>::zeros((number_of_clusters, number_of_features));
            dense
                .rows()
                .into_iter()
                .zip(assignments.iter())
                .for_each(|(row, &cluster)| {
                    let mut sum_row = batch_sums.row_mut(cluster);
                    azip!((sum in &mut sum_row, &value in &row) *sum = *sum + value);
                });

            centers
                .rows_mut()
                .into_iter()
                .zip(counts.iter_mut())
                .zip(batch_sums.rows().into_iter().zip(batch_counts.iter()))
                .for_each(|((mut center_row, count), (sum_row, &batch_count))| {
                    if batch_count > 0 {
                        let prior = F::from(*count).expect("row count fits scalar");
                        let updated_count = *count + batch_count;
                        let total = F::from(updated_count).expect("row count fits scalar");
                        let prior_weight = prior / total;
                        let inverse_total = F::one() / total;
                        azip!((center in &mut center_row, &sum in &sum_row) {
                            *center = *center * prior_weight + sum * inverse_total;
                        });
                        *count = updated_count;
                    }
                });
        }
        SKDataView::Sparse(sparse) => {
            if !sparse.is_csr() {
                return Err(SKError::UnsupportedRepresentation {
                    representation: "csc",
                    suggestion: "convert the sparse matrix to CSR before calling MiniBatchKMeans",
                });
            }
            let mut inverse_totals = vec![F::zero(); number_of_clusters];
            centers
                .rows_mut()
                .into_iter()
                .zip(counts.iter_mut())
                .zip(batch_counts.iter().zip(inverse_totals.iter_mut()))
                .for_each(|((mut center_row, count), (&batch_count, inverse_slot))| {
                    if batch_count > 0 {
                        let prior = F::from(*count).expect("row count fits scalar");
                        let updated_count = *count + batch_count;
                        let total = F::from(updated_count).expect("row count fits scalar");
                        let prior_weight = prior / total;
                        *inverse_slot = F::one() / total;
                        center_row.mapv_inplace(|value| value * prior_weight);
                        *count = updated_count;
                    }
                });

            sparse
                .outer_iterator()
                .zip(assignments.iter())
                .for_each(|(stored_row, &cluster)| {
                    let scale = inverse_totals[cluster];
                    let mut center_row = centers.row_mut(cluster);
                    stored_row.iter().for_each(|(column, &value)| {
                        center_row[column] = center_row[column] + value * scale;
                    });
                });
        }
        _ => {
            return Err(SKError::UnsupportedRepresentation {
                representation: features.representation(),
                suggestion: "convert the input to a supported dense or CSR view",
            });
        }
    }

    let maximum_shift = previous_centers
        .rows()
        .into_iter()
        .zip(centers.rows().into_iter())
        .fold(F::zero(), |best, (old_row, new_row)| {
            let shift = dense_squared_distance(old_row, new_row).sqrt();
            if shift > best { shift } else { best }
        });
    Ok(maximum_shift)
}
