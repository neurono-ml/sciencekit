//! Silhouette samples and the mean silhouette score (spec `clustering-metrics`).

use ndarray::{Array1, ArrayView1, s};
use rayon::prelude::*;
use sciencekit_common::{SKClusterLabel, SKDataView, SKError, SKFloat};
use sciencekit_math::{SKPairwiseMetric, sk_pairwise_distance_matrix};

use super::grouping::{SKClusterGrouping, sk_group_clusters};
use super::inputs::sk_validate_clustering_inputs;

/// Rows processed per distance chunk; bounds the working matrix to
/// `ROW_CHUNK × number_of_rows` distances instead of a full `n × n` matrix.
const ROW_CHUNK: usize = 128;

/// Per-sample silhouette values for every non-noise row.
///
/// The sample value is `(b − a) / max(a, b)`, where `a` is the mean distance to
/// the row's own cluster and `b` the minimum mean distance to another cluster.
/// Distances are Euclidean, computed chunk by chunk through the pairwise
/// dispatcher, so dense and CSR input share the same path without densifying.
///
/// **Noise policy:** noise rows are excluded; the result carries one value per
/// non-noise row, in row order.
pub fn sk_silhouette_samples<F: SKFloat>(
    features: SKDataView<'_, F>,
    assignments: ArrayView1<'_, SKClusterLabel>,
) -> Result<Array1<F>, SKError> {
    let (number_of_rows, _) = sk_validate_clustering_inputs(&features, assignments, "silhouette")?;
    let grouping = sk_group_clusters(assignments);
    if grouping.number_of_clusters() == 0 {
        return Err(SKError::EmptyInput {
            operation: "silhouette",
        });
    }
    if grouping.number_of_clusters() < 2 {
        return Err(SKError::InvalidHyperparameter {
            name: "assignments",
            reason: "silhouette requires at least two clusters after excluding noise".to_string(),
        });
    }

    let mut positions = vec![None; number_of_rows];
    let mut number_of_samples = 0;
    for (row, cluster) in grouping.row_cluster.iter().enumerate() {
        if cluster.is_some() {
            positions[row] = Some(number_of_samples);
            number_of_samples += 1;
        }
    }

    let cluster_sizes: Vec<F> = grouping
        .cluster_sizes
        .iter()
        .map(|size| F::from(*size).expect("a cluster size fits in the scalar"))
        .collect();
    let chunks: Vec<(usize, usize)> = (0..number_of_rows)
        .step_by(ROW_CHUNK)
        .map(|start| (start, (start + ROW_CHUNK).min(number_of_rows)))
        .collect();
    let chunk_values = chunks
        .par_iter()
        .map(|&(start, end)| {
            sk_chunk_samples(features, &grouping, &positions, &cluster_sizes, start, end)
        })
        .collect::<Result<Vec<_>, SKError>>()?;

    let mut samples = Array1::<F>::zeros(number_of_samples);
    for values in chunk_values {
        for (position, value) in values {
            samples[position] = value;
        }
    }
    Ok(samples)
}

/// Mean silhouette score over every non-noise row.
///
/// Equivalent to the mean of [`sk_silhouette_samples`], with the same noise
/// policy and the same rejection of fewer than two clusters.
pub fn sk_silhouette_score<F: SKFloat>(
    features: SKDataView<'_, F>,
    assignments: ArrayView1<'_, SKClusterLabel>,
) -> Result<F, SKError> {
    let samples = sk_silhouette_samples(features, assignments)?;
    let count = F::from(samples.len()).expect("the sample count fits in the scalar");
    let total = samples
        .iter()
        .fold(F::zero(), |accumulated, value| accumulated + *value);
    Ok(total / count)
}

/// Silhouette values of one row chunk against every row.
fn sk_chunk_samples<F: SKFloat>(
    features: SKDataView<'_, F>,
    grouping: &SKClusterGrouping,
    positions: &[Option<usize>],
    cluster_sizes: &[F],
    start: usize,
    end: usize,
) -> Result<Vec<(usize, F)>, SKError> {
    let chunk = sk_row_chunk(&features, start, end);
    let distances = sk_pairwise_distance_matrix(SKPairwiseMetric::Euclidean, chunk, features)?;
    let number_of_clusters = grouping.number_of_clusters();
    let mut cluster_sums = vec![F::zero(); number_of_clusters];
    let mut values = Vec::with_capacity(end - start);
    for (local_row, distance_row) in distances.rows().into_iter().enumerate() {
        let row = start + local_row;
        let Some(own) = grouping.row_cluster[row] else {
            continue;
        };
        for sum in cluster_sums.iter_mut() {
            *sum = F::zero();
        }
        for (column, distance) in distance_row.iter().enumerate() {
            if let Some(cluster) = grouping.row_cluster[column] {
                cluster_sums[cluster] = cluster_sums[cluster] + *distance;
            }
        }
        let value = sk_sample_value(own, &cluster_sums, cluster_sizes);
        values.push((
            positions[row].expect("every non-noise row has a sample position"),
            value,
        ));
    }
    Ok(values)
}

/// The silhouette value of one row from its per-cluster distance sums.
fn sk_sample_value<F: SKFloat>(own: usize, cluster_sums: &[F], cluster_sizes: &[F]) -> F {
    let own_size = cluster_sizes[own];
    if own_size <= F::one() {
        return F::zero();
    }
    let own_mean = cluster_sums[own] / (own_size - F::one());
    let mut nearest_mean = F::infinity();
    for (cluster, sum) in cluster_sums.iter().enumerate() {
        if cluster != own {
            let mean = *sum / cluster_sizes[cluster];
            if mean < nearest_mean {
                nearest_mean = mean;
            }
        }
    }
    let larger = if own_mean > nearest_mean {
        own_mean
    } else {
        nearest_mean
    };
    if larger.is_zero() {
        F::zero()
    } else {
        (nearest_mean - own_mean) / larger
    }
}

/// Borrow one contiguous row chunk of either representation.
fn sk_row_chunk<'view, F>(
    features: &'view SKDataView<'_, F>,
    start: usize,
    end: usize,
) -> SKDataView<'view, F> {
    match features {
        SKDataView::Dense(dense) => SKDataView::Dense(dense.slice(s![start..end, ..])),
        SKDataView::Sparse(sparse) => SKDataView::Sparse(sparse.slice_outer_rbr(start..end)),
        _ => unreachable!("SKDataView only exposes dense and csr"),
    }
}
