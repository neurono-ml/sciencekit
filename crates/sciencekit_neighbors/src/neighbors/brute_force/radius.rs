//! The chunked, parallel radius batch of the brute-force searcher.

use ndarray::ArrayView1;
use rayon::prelude::*;
use sciencekit_common::{SKDataView, SKError, SKFloat};
use sciencekit_math::pairwise::sk_pairwise_distance_matrix;

use super::super::SKNeighborsMetric;
use super::super::validation::{sk_validate_finiteness, sk_view_dimensions};
use super::k_nearest::{
    SK_PAIRWISE_CHUNK_BUDGET_BYTES, sk_chunk_rows, sk_pair_order, sk_query_chunk,
};
use crate::searcher::SKRadiusNeighbors;
use crate::searcher::candidates::sk_should_run_parallel;

/// Every training row within the inclusive `radius` of each query row.
pub(crate) fn sk_brute_force_radius_neighbors<F: SKFloat>(
    training: SKDataView<'_, F>,
    metric: SKNeighborsMetric,
    queries: SKDataView<'_, F>,
    radius: F,
) -> Result<SKRadiusNeighbors<F>, SKError> {
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
            operation: "radius_neighbors",
        });
    }
    if !radius.is_finite() || radius < F::zero() {
        return Err(SKError::InvalidHyperparameter {
            name: "radius",
            reason: "must be finite and non-negative".to_string(),
        });
    }
    sk_validate_finiteness(training)?;
    sk_validate_finiteness(queries)?;

    let chunk_rows = sk_chunk_rows::<F>(SK_PAIRWISE_CHUNK_BUDGET_BYTES, training_rows, query_rows);
    let parallel = sk_should_run_parallel(query_rows, training_rows);
    let mut distances: Vec<Vec<F>> = Vec::with_capacity(query_rows);
    let mut indices: Vec<Vec<usize>> = Vec::with_capacity(query_rows);

    for start in (0..query_rows).step_by(chunk_rows) {
        let end = (start + chunk_rows).min(query_rows);
        let matrix = sk_pairwise_distance_matrix(
            metric.pairwise_metric(),
            sk_query_chunk(&queries, start, end)?,
            training,
        )?;
        let rows = matrix.nrows();
        let select_row = |row: usize| sk_within_radius(matrix.row(row), radius);
        let selections: Vec<Vec<(F, usize)>> = if parallel {
            (0..rows).into_par_iter().map(select_row).collect()
        } else {
            (0..rows).map(select_row).collect()
        };
        for selection in selections {
            distances.push(selection.iter().map(|(distance, _)| *distance).collect());
            indices.push(selection.iter().map(|(_, index)| *index).collect());
        }
    }

    Ok(SKRadiusNeighbors::new(distances, indices))
}

/// Collect the inclusive-radius candidates of one row, ordered deterministically.
fn sk_within_radius<F: SKFloat>(distances: ArrayView1<'_, F>, radius: F) -> Vec<(F, usize)> {
    let mut candidates: Vec<(F, usize)> = distances
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, distance)| *distance <= radius)
        .map(|(index, distance)| (distance, index))
        .collect();
    candidates.sort_unstable_by(sk_pair_order);
    candidates
}
