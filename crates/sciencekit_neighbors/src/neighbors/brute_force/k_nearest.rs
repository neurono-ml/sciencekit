//! The chunked, parallel `k`-nearest batch of the brute-force searcher.

use ndarray::{Array2, ArrayView1, Axis, Slice};
use rayon::prelude::*;
use sciencekit_common::{SKDataView, SKError, SKFloat};
use sciencekit_math::pairwise::sk_pairwise_distance_matrix;

use super::super::SKNeighborsMetric;
use super::super::validation::{
    sk_validate_finiteness, sk_validate_neighbors_count, sk_view_dimensions,
};
use crate::searcher::SKNeighborQuery;
use crate::searcher::candidates::sk_should_run_parallel;

/// Memory budget of one dense distance chunk: a query batch never
/// materializes the whole `queries × training` matrix at once.
pub(crate) const SK_PAIRWISE_CHUNK_BUDGET_BYTES: usize = 32 * 1024 * 1024;

/// Compute the exact `k` nearest training rows per query row.
pub(crate) fn sk_brute_force_k_nearest<F: SKFloat>(
    training: SKDataView<'_, F>,
    metric: SKNeighborsMetric,
    queries: SKDataView<'_, F>,
    neighbors_count: usize,
) -> Result<SKNeighborQuery<F>, SKError> {
    sk_brute_force_k_nearest_with_budget(
        training,
        metric,
        queries,
        neighbors_count,
        SK_PAIRWISE_CHUNK_BUDGET_BYTES,
    )
}

/// The budget-parameterized form, so tests can force many small chunks.
pub(crate) fn sk_brute_force_k_nearest_with_budget<F: SKFloat>(
    training: SKDataView<'_, F>,
    metric: SKNeighborsMetric,
    queries: SKDataView<'_, F>,
    neighbors_count: usize,
    budget_bytes: usize,
) -> Result<SKNeighborQuery<F>, SKError> {
    let (training_rows, training_columns) = sk_view_dimensions(&training);
    let (query_rows, query_columns) = sk_view_dimensions(&queries);
    if query_columns != training_columns {
        return Err(SKError::shape_mismatch_2d(
            query_rows,
            training_columns,
            query_rows,
            query_columns,
        ));
    }
    if query_rows == 0 {
        return Err(SKError::EmptyInput {
            operation: "k_nearest",
        });
    }
    sk_validate_finiteness(training)?;
    sk_validate_finiteness(queries)?;
    sk_validate_neighbors_count(neighbors_count, training_rows, "k_nearest")?;

    let chunk_rows = sk_chunk_rows::<F>(budget_bytes, training_rows, query_rows);
    let parallel = sk_should_run_parallel(query_rows, training_rows);
    let mut distances = Array2::zeros((query_rows, neighbors_count));
    let mut indices = Array2::zeros((query_rows, neighbors_count));

    for start in (0..query_rows).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(query_rows);
        let matrix = sk_pairwise_distance_matrix(
            metric.pairwise_metric(),
            sk_query_chunk(&queries, start, end)?,
            training,
        )?;
        let rows = matrix.nrows();
        let select_row = |row: usize| sk_select_k(matrix.row(row), neighbors_count);
        let selections: Vec<Vec<(F, usize)>> = if parallel {
            (0..rows).into_par_iter().map(select_row).collect()
        } else {
            (0..rows).map(select_row).collect()
        };
        for (offset, selection) in selections.into_iter().enumerate() {
            for (column, (distance, index)) in selection.into_iter().enumerate() {
                distances[[start + offset, column]] = distance;
                indices[[start + offset, column]] = index;
            }
        }
    }

    Ok(SKNeighborQuery::new(distances, indices))
}

/// The number of query rows one chunk may hold under the byte budget.
pub(crate) fn sk_chunk_rows<F: SKFloat>(
    budget_bytes: usize,
    training_rows: usize,
    query_rows: usize,
) -> usize {
    let bytes_per_query_row = training_rows
        .saturating_mul(std::mem::size_of::<F>())
        .max(1);
    (budget_bytes / bytes_per_query_row)
        .max(1)
        .min(query_rows.max(1))
}

/// A representation-preserving slice of query rows.
pub(super) fn sk_query_chunk<'a, 'b, F>(
    queries: &'b SKDataView<'a, F>,
    start: usize,
    end: usize,
) -> Result<SKDataView<'b, F>, SKError>
where
    'a: 'b,
{
    match queries {
        SKDataView::Dense(dense) => Ok(SKDataView::Dense(
            dense.slice_axis(Axis(0), Slice::from(start..end)),
        )),
        SKDataView::Sparse(sparse) => Ok(SKDataView::Sparse(sparse.slice_outer(start..end))),
        other => Err(SKError::UnsupportedRepresentation {
            representation: other.representation(),
            suggestion: "convert the input to a dense matrix or a CSR matrix",
        }),
    }
}

/// Select the `k` smallest `(distance, index)` pairs of one distance row.
fn sk_select_k<F: SKFloat>(
    distances: ArrayView1<'_, F>,
    neighbors_count: usize,
) -> Vec<(F, usize)> {
    let mut candidates: Vec<(F, usize)> = distances
        .iter()
        .copied()
        .enumerate()
        .map(|(index, distance)| (distance, index))
        .collect();
    if neighbors_count < candidates.len() {
        candidates.select_nth_unstable_by(neighbors_count - 1, sk_pair_order);
        candidates.truncate(neighbors_count);
    }
    candidates.sort_unstable_by(sk_pair_order);
    candidates
}

/// Distance ascending, then the smallest training index.
pub(super) fn sk_pair_order<F: SKFloat>(
    left: &(F, usize),
    right: &(F, usize),
) -> std::cmp::Ordering {
    left.0
        .partial_cmp(&right.0)
        .unwrap_or(std::cmp::Ordering::Equal)
        .then_with(|| left.1.cmp(&right.1))
}
